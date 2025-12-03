import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import time

model_path = "roberta_wwm_mlm_dynamic_new.onnx"
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

opt = ort.SessionOptions()
opt.intra_op_num_threads = 8
opt.inter_op_num_threads = 8

# fast CPU/GPU execution provider
session = ort.InferenceSession(
    model_path,
    sess_options=opt,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)


def mlm_ppl_batch(text):
    tokens = tokenizer(text, return_tensors="np")
    input_ids = tokens["input_ids"][0].astype("int64")
    attention_mask = tokens["attention_mask"][0].astype("int64")

    seq_len = input_ids.shape[0]
    mask_pos = list(range(1, seq_len - 1))
    N = len(mask_pos)

    batch_input = np.tile(input_ids, (N, 1))
    for i, pos in enumerate(mask_pos):
        batch_input[i, pos] = tokenizer.mask_token_id

    batch_att = np.tile(attention_mask, (N, 1))
    token_type_ids = np.zeros_like(batch_input, dtype="int64")

    outputs = session.run(
        ["logits"],
        {
            "input_ids": batch_input,
            "attention_mask": batch_att,
            "token_type_ids": token_type_ids,
        }
    )[0]

    # compute MLM loss
    losses = []
    for i, pos in enumerate(mask_pos):
        logits = outputs[i, pos]
        t = input_ids[pos]
        log_probs = logits - logits.max()  # stable softmax
        probs = np.exp(log_probs)
        losses.append(-np.log(probs[t] / probs.sum() + 1e-12))

    ppl = np.exp(np.mean(losses))
    return ppl


def is_meaningful(text, threshold=50):
    ppl = mlm_ppl_batch(text)
    return ppl < threshold, ppl


# ---------------- TEST ----------------


samples = [
    "我今天去了广州出差。",
    "阿斯蒂芬家家都发啊阿凡达的法案",
    "上海市徐汇区虹桥路500号。",
    "asfjaiosjfioasjfioasjfioasjdfiojasdf",
    "持续营造风清气正的网络空间，推动构建网络空间命运共同体，总书记关心的这件事，和屏幕前的你我息息相关。",
    "吗的时间覅放大假 撒嗲副驾驶法发啊啊打发撒",
    "啊的手法首发卡的发觉啊打发十分的",
    "这个产品的功能非常全面，操作也很简单",
    "asdfaa asfda ;joijoi  asdfad",
    "从网友发布的照片来看，新工服上身为红色基调，胸口和背后有明显的京东外卖品牌logo，下身为灰色裤子，整体设计风格与“法拉利”赛车服相似。目前京东外卖员的工服虽同样以红色为底色，但设计相对简单，主要搭配金色条状装饰。",    "和阿里一样，今年以来，京东在推广外卖业务方面来势汹汹。早在3月，京东正式入局外卖领域，明确将自身定位为“品质外卖”。上线初期，京东便迅速在全国39个城市开展业务，吸引了近20万餐饮商家申请入驻。凭借已有的用户基础和品牌影响力，当月24日，京东宣布外卖日单量突破100万单。4月，京东在外卖领域的动作更加频繁，例如通过一系列优惠政策和补贴活动吸引了大量用户，同时加大全职骑手的招聘力度，招收不低于五万名全职外卖员。4月21日，京东又将未来三个月外卖员的招聘名额提高到十万名。"
]

for s in samples:
    start_time = time.time()
    print(s, is_meaningful(s))

    end_time = time.time()
    print(f"Batch版耗时: {end_time - start_time:.6f} 秒")

