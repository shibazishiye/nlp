from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import math
import time


model_name = "hfl/chinese-roberta-wwm-ext"

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.set_num_threads(8)
torch.set_num_interop_threads(8)


@torch.no_grad()
def sentence_score_batch(text):
    """
    Batch MLM perplexity using Chinese-RoBERTa-wwm-ext
    O(N) instead of original O(N^2)
    """
    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"][0].to(device)
    attention_mask = enc["attention_mask"][0].to(device)

    seq_len = input_ids.size(0)

    # 忽略特殊 token：不 mask CLS(0) 和 SEP(-1)
    positions = list(range(1, seq_len - 1))
    num_samples = len(positions)

    # 批次：每个位置一个 masked 样本
    batch_input = input_ids.unsqueeze(0).repeat(num_samples, 1)
    for n, pos in enumerate(positions):
        batch_input[n, pos] = tok.mask_token_id

    batch_att = attention_mask.unsqueeze(0).repeat(num_samples, 1)

    outputs = model(
        input_ids=batch_input,
        attention_mask=batch_att
    ).logits  # shape = [num_samples, seq_len, vocab]

    # 计算 loss
    losses = []
    for idx, pos in enumerate(positions):
        logits = outputs[idx, pos]
        target_id = input_ids[pos]
        loss = -torch.log_softmax(logits, dim=-1)[target_id]
        losses.append(loss)

    loss = torch.stack(losses).mean()
    ppl = torch.exp(loss)
    return ppl.item()


def is_meaningful(text, threshold=50):
    ppl = sentence_score_batch(text)
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
