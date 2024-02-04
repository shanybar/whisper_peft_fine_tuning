import os
import gc
import torch
import evaluate
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from peft import prepare_model_for_int8_training
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperForConditionalGeneration
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import WhisperTokenizer, WhisperProcessor, WhisperFeatureExtractor
from peft import PeftModel, LoraConfig, get_peft_model, PeftConfig, prepare_model_for_kbit_training
from src.data.data_processing import load_data, prepare_dataset, DataCollatorSpeechSeq2SeqWithPadding
from transformers import Seq2SeqTrainer, TrainerCallback,\
    TrainingArguments, TrainerState, TrainerControl, BitsAndBytesConfig


class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


def compute_metrics(pred, tokenizer, metric):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def train_model(model_name_or_path, speech_data, data_collator, processor, tokenizer, metric):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, quantization_config=bnb_config,
                                                            device_map='auto')
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(r=16, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir="training_output",  # change to a repo name of your choice
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-4,
        warmup_steps=100,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        fp16=True,
        per_device_eval_batch_size=4,
        # the argument below can't be passed to Trainer because it internally calls transformer's generate without autocasting (to int8) leading to errors
        predict_with_generate=True,
        generation_max_length=128,
        logging_steps=25,
        remove_unused_columns=False,
        # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
        optim="paged_adamw_8bit"
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=speech_data["train"],
        eval_dataset=speech_data["test"],
        data_collator=data_collator,
        compute_metrics=lambda n: compute_metrics(n, metric=metric,tokenizer=tokenizer),
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback],
    )
    model.config.use_cache = False

    trainer.train()


def eval_model(speech_data, data_collator, tokenizer, metric):
    peft_model_id = "" # TBD
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, peft_model_id)

    eval_dataloader = DataLoader(speech_data["test"], batch_size=8, collate_fn=data_collator)

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                        max_new_tokens=255,
                    )
                        .cpu()
                        .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
        del generated_tokens, labels, batch
        gc.collect()
    wer = 100 * metric.compute()
    print(f"{wer=}")


def train_and_eval_model():
    model_name_or_path = "openai/whisper-base"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    lang_name = "Greek"
    lang_short = 'el'  # Greek
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=lang_name, task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=lang_name, task="transcribe")

    common_voice = load_data(lang_short)

    common_voice = common_voice.map(lambda n: prepare_dataset(n, feature_extractor=feature_extractor,
                                    tokenizer=tokenizer),
                                    remove_columns=common_voice.column_names["train"])

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

    train_model(model_name_or_path, common_voice, data_collator, processor, tokenizer, metric)
    # eval_model(common_voice, data_collator, tokenizer, metric)

if __name__ == '__main__':
    train_and_eval_model()






