"""CLI entry-point for RAG-based SFT code generation."""
import argparse
from ab.gpt.util.nn_sftcodegen_rag import main as sft_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NN architectures with finetuned LLM + RAG")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Generation epochs (default: 1)")
    parser.add_argument("-c", "--config", type=str, default="nngpt_unique_arch_rag.json", 
                        help="Config JSON in conf/llm/ (default: nngpt_unique_arch_rag.json)")
    parser.add_argument("-n", "--max_items", type=int, default=300, 
                        help="Max items from NN_Rag_gen_test.jsonl (default: 300)")
    parser.add_argument("-p", "--prompt_template", type=str, default="unique_rag_test_rules.json",
                        help="Prompt template in conf/prompt/ (default: unique_rag_test_rules.json)")
    parser.add_argument("--temperature", type=float, default=0.20, 
                        help="Generation temperature (default: 0.20)")
    parser.add_argument("--no_rejection_sampling", action="store_true",
                        help="Disable novelty/known-block rejection")

    args = parser.parse_args()



    sft_main(
        epoch=args.epochs - 1,
        config=args.config,
        max_items=args.max_items,
        prompt_template=args.prompt_template,
        temperature=args.temperature,
        rejection_sampling=not args.no_rejection_sampling,
    )


if __name__ == "__main__":
    main()
