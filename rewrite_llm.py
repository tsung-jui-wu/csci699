import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROMPT_FILE = Path(__file__).parent / "abstract_rewrite_prompt_1.txt"


def rewrite_abstract(abstract: str, model: str = "gpt-4o") -> str:
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")
    prompt = prompt_template.replace("[PASTE YOUR ABSTRACT HERE]", abstract)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content

# # On
# abs = "Self-supervised learning (SSL) approaches have brought tremendous success across many tasks and domains. It has been argued that these successes can be attributed to a link between SSL and identifiable representation learning: Temporal structure and auxiliary variables ensure that latent representations are related to the true un- derlying generative factors of the data. Here, we deepen this connection and show that SSL can perform system identification in latent space. We propose dynamics contrastive learning, a framework to uncover linear, switching linear and non-linear dynamics under a non-linear observation model, give theoretical guarantees and val- idate them empirically."
# paper_path = "D:\csci699\data\pdfs\ONfWFluZBI.pdf"

# #Pw
# abs = "Graph Convolutional Networks (GCNs) have emerged as powerful tools for learn- ing on graph-structured data, yet the behavior of dropout in these models re- mains poorly understood. This paper presents a comprehensive theoretical analy- sis of dropout in GCNs, revealing that its primary role differs fundamentally from standard neural networks - preventing oversmoothing rather than co-adaptation. We demonstrate that dropout in GCNs creates dimension-specific stochastic sub- graphs, leading to a form of structural regularization not present in standard neural networks. Our analysis shows that dropout effects are inherently degree- dependent, resulting in adaptive regularization that considers the topological im- portance of nodes. We provide new insights into dropout’s role in mitigating oversmoothing and derive novel generalization bounds that account for graph- specific dropout effects. Furthermore, we analyze the synergistic interaction be- tween dropout and batch normalization in GCNs, uncovering a mechanism that enhances overall regularization. Our theoretical findings are validated through ex- tensive experiments on both node-level and graph-level tasks across 14 datasets. Notably, GCN with dropout and batch normalization outperforms state-of-the-art methods on several benchmarks, demonstrating the practical impact of our theo- retical insights"
# paper_path = "D:\csci699\data\pdfs\PwxYoMvmvy.pdf"

# #vi
# abs = "As language models advance, traditional benchmarks face challenges of dataset saturation and disconnection from real-world performance, limiting our under- standing of true model capabilities. We introduce EXecution-Eval (EXE), a benchmark designed to assess LLMs’ ability to execute code and predict program states. EXE attempts to address key limitations in existing evaluations: difficulty scaling, task diversity, training data contamination, and cost-effective scalability. Comprising over 30,000 tasks derived from 1,000 popular Python repositories on GitHub, EXE spans a wide range of lengths and algorithmic complexities. Tasks require models to execute code, necessitating various operations including math- ematical reasoning, logical inference, bit manipulation, string operations, loop execution, and maintaining multiple internal variable states during computation. Our methodology involves: (a) selecting and preprocessing GitHub repositories, (b) generating diverse inputs for functions, (c) executing code to obtain ground truth outputs, and (d) formulating tasks that require models to reason about code execution. This approach allows for continuous new task generation for as few as 1,123 tokens, significantly reducing the risk of models ”training on the test set.” We evaluate several state-of-the-art LLMs on EXE, revealing insights into their code comprehension and execution capabilities. Our results show that even the best-performing models struggle with complex, multi-step execution tasks, highlighting specific computational concepts that pose the greatest challenges for today’s LLMs. Furthermore, we review EXE’s potential for finding and predicting errors to aid in assessing a model’s cybersecurity capabilities. We propose EXE as a sustainable and challenging testbed for evaluating frontier models, offering insights into their internal mechanistic advancement."
# paper_path = "D:\csci699\data\pdfs\viQ1bLqKY0.pdf"

#zk
abs = "Information retrieval across different languages is an increasingly important chal- lenge in natural language processing. Recent approaches based on multilingual pre-trained language models have achieved remarkable success, yet they often optimize for either monolingual, cross-lingual, or multilingual retrieval perfor- mance at the expense of others. This paper proposes a novel hybrid batch training strategy to simultaneously improve zero-shot retrieval performance across mono- lingual, cross-lingual, and multilingual settings while mitigating language bias. The approach fine-tunes multilingual language models using a mix of monolingual and cross-lingual question-answer pair batches sampled based on dataset size. Experiments on XQuAD-R, MLQA-R, and MIRACL benchmark datasets show that the proposed method consistently achieves comparable or superior results in zero-shot retrieval across various languages and retrieval tasks compared to monolingual-only or cross-lingual-only training. Hybrid batch training also sub- stantially reduces language bias in multilingual retrieval compared to monolingual training. These results demonstrate the effectiveness of the proposed approach for learning language-agnostic representations that enable strong zero-shot retrieval performance across diverse languages"
paper_path = "D:\csci699\data\pdfs\zkNCWtw2fd.pdf"
def main():
    parser = argparse.ArgumentParser(description="Rewrite a paper abstract using GPT.")
    parser.add_argument("--abstract", default=abs, help="The original abstract text to rewrite")
    parser.add_argument("--paper_path",default=paper_path, help="Path to the paper PDF (for reference)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    args = parser.parse_args()

    print(f"Paper: {args.paper_path}")
    print(f"Model: {args.model}\n")
    print("=" * 60)

    result = rewrite_abstract(args.abstract, model=args.model)
    print(result)


if __name__ == "__main__":
    main()
