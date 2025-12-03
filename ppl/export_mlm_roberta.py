import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

model.eval()
model.cpu()

# create a batch=2 dummy to force correct dynamic axis
enc = tokenizer(["用于 ONNX 动态测试", "第二个样本"], 
                return_tensors="pt", 
                padding="max_length", 
                max_length=32)

dummy_ids = enc["input_ids"]
dummy_att = enc["attention_mask"]
dummy_token = torch.zeros_like(dummy_ids)

torch.onnx.export(
    model,
    (dummy_ids, dummy_att, dummy_token),
    "roberta_wwm_mlm_dynamic_new.onnx",
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    opset_version=13,
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "token_type_ids": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
    }
)

print("DONE: roberta_wwm_mlm_dynamic_new.onnx")

