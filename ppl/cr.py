from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import math
import time

model_name = "hfl/chinese-roberta-wwm-ext"

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

def sentence_score(text):
    # 使用 MLM 做困惑度估计
    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]

    loss = 0
    count = 0

    for i in range(1, len(input_ids)-1):
        masked = input_ids.clone()
        masked[i] = tok.mask_token_id

        out = model(masked.unsqueeze(0))
        logits = out.logits[0, i]
        prob = torch.softmax(logits, dim=-1)
        target_prob = prob[input_ids[i]]
        loss += -torch.log(target_prob + 1e-12)
        count += 1

    ppl = torch.exp(loss / count)
    return ppl.item()   # 越低越自然

def is_meaningful(text, threshold=50):
    ppl = sentence_score(text)
    return ppl < threshold, ppl

# 测试



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
    elapsed_time = end_time - start_time
    print(f"简单计时, 耗时: {elapsed_time:.6f} 秒")