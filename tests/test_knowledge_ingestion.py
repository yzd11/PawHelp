"""Knowledge document ingestion through its public chunking seam."""

from pathlib import Path

from app.knowledge.chunking import chunk_markdown


def test_markdown_becomes_self_contained_structured_knowledge_chunks() -> None:
    markdown = """# 售后政策

## 退货条件

第一句话说明适用范围。第二句话说明商品状态。第三句话说明申请方式。第四句话说明审核结果。

> [!IMPORTANT]
> 超过规定期限后不能直接申请退货。

## 退款进度

| 支付方式 | 到账时间 |
| --- | --- |
| 微信 | 1 至 3 天 |
| 支付宝 | 1 至 3 天 |
| 银行卡 | 3 至 7 天 |
"""

    chunks = chunk_markdown(
        markdown,
        content_type="policy",
        chunk_size=34,
        chunk_overlap=12,
    )

    return_chunks = [chunk for chunk in chunks if chunk.questions == "退货条件"]
    assert return_chunks
    assert all(chunk.category == "售后政策" for chunk in return_chunks)
    assert all(chunk.section_path == "售后政策 / 退货条件" for chunk in return_chunks)
    assert any(chunk.is_key_clause for chunk in return_chunks)
    assert all(not chunk.answer.startswith("句话") for chunk in return_chunks[1:])

    table_chunks = [chunk for chunk in chunks if "| 支付方式 | 到账时间 |" in chunk.answer]
    assert len(table_chunks) >= 2
    assert all("| --- | --- |" in chunk.answer for chunk in table_chunks)
    assert sum("| 支付方式 | 到账时间 |" in chunk.answer for chunk in table_chunks) == len(
        table_chunks
    )


def test_every_manual_section_names_the_products_it_covers() -> None:
    # A section chunk is checked alone; without the product series a model-specific question is refused.
    manuals = Path(__file__).parents[1] / "knowledge" / "manuals"
    for name in ("feeder-guide", "fountain-care", "litter-box-care", "camera-guide"):
        path = manuals / f"{name}.md"
        chunks = chunk_markdown(path.read_text(encoding="utf-8"), content_type="manual")
        assert chunks and all("MH-" in chunk.category for chunk in chunks), path.name
