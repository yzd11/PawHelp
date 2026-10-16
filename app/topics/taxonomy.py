"""Authoritative 17-topic taxonomy for offline flywheel classification."""
from dataclasses import dataclass, asdict
import hashlib
import json


@dataclass(frozen=True)
class Topic:
    name: str
    boundary: str
    example: str


TOPICS = (
    Topic("退换货", "退货、退款、普通换货诉求；维修保修流程中的故障换新归保修维修。", "买大了想退；换小一号"),
    Topic("物流", "发货、运输、签收、丢件和配送时效；运费的钱事归运费。", "快递到哪了；补发几天到"),
    Topic("尺码", "大小、码数及穿戴/容纳适配；颗粒、容量等商品规格归商品信息。", "猫窝买大了；项圈太紧"),
    Topic("发票", "开票、抬头、税号及报销凭证。", "发票抬头能改吗"),
    Topic("质量问题", "明确陈述商品破损、故障、缺陷；不因退货/维修诉求脑补质量问题。", "饮水机漏水；爬架断裂"),
    Topic("运费", "运费金额、承担方、包邮及运费险理赔；货走到哪归物流。", "退货运费谁出"),
    Topic("优惠活动", "优惠券、满减、限时折扣及叠加活动；会员等级权益归会员积分，购买后补差归价保。", "券能和满减叠加吗"),
    Topic("价保", "购买后降价的价格保护、补差价及申请时效；下单优惠归优惠活动。", "刚买就降价能补差吗"),
    Topic("支付", "付款方式、支付失败、扣款及分期；提到退款到账不自动加支付。", "扣了两次钱；支持分期吗"),
    Topic("订单修改", "下单后修改地址/联系方式/商品或取消订单；询问发货归物流。", "订单地址填错了"),
    Topic("库存补货", "是否有货、库存和补货时间；不因假设补货后降价就自动加此类。", "啥时候补货"),
    Topic("商品信息", "材质、功能、用法、保养、规格和商品回收处理说明；穿戴适配归尺码。", "猫窝能机洗吗；旧猫抓板怎么回收"),
    Topic("保修维修", "保修期限、维修服务、零件补换及保修内故障换新；普通退换归退换货。", "保修多久；坏了能修吗"),
    Topic("账号", "登录、密码、绑定、注销和账号安全。", "登录不上；怎么换绑手机号"),
    Topic("会员积分", "会员等级、会员权益、积分获得及使用。", "会员专享价几级解锁；积分抵运费吗"),
    Topic("评价", "评价、追评、晒单及其规则。", "评价能改吗"),
    Topic("其他", "整句不匹配前16类时兜底，不与具体主题共存；否定/背景提及不等于有效诉求。", "你好；转人工；客服几点上班"),
)
TOPIC_NAMES = tuple(topic.name for topic in TOPICS)
LABEL2ID = {name: i for i, name in enumerate(TOPIC_NAMES)}
ID2LABEL = dict(enumerate(TOPIC_NAMES))
TAXONOMY_HASH = hashlib.sha256(json.dumps([asdict(t) for t in TOPICS], ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_labels(labels: object) -> list[str]:
    if not isinstance(labels, list) or not labels or any(not isinstance(x, str) or x not in LABEL2ID for x in labels):
        raise ValueError("labels must be a nonempty array of authoritative topic names")
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate labels")
    if "其他" in labels and len(labels) != 1:
        raise ValueError("其他 is an exclusive whole-sentence fallback")
    return sorted(labels, key=LABEL2ID.__getitem__)


def terminology_table() -> str:
    return "\n".join(f"{t.name}：{t.boundary} 例：{t.example}" for t in TOPICS)
