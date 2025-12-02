
# Caissa (nanoGPT fork)

Chess-based LLM trained exclusively on position + best move pairs without game context. Aims to explore the limits of natural-language-based processing for chess tasks along with searchless evaluation.  

## Installation and Dependencies

```
# For .\train-caissa and .\caissa-v.0
pip install torch numpy transformers datasets tiktoken wandb tqdm
# For .\evaluate-caissa and .\create-train-val-puzzle-sets
pip install 
```

Dependencies for .\train-caissa and .\caissa-v.0:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

Dependencies for .\evaluate-caissa and .\create-train-val-puzzle-sets:

- [pytorch](https://pytorch.org) <3

## Dataset (.\create-train-val-puzzle-sets)

## Training (.\train-caissa)

All code in the .\train-caissa directory (which was used to train Caissa [duh!]) was initially forked from Andrej Karpathy's nanoGPT repository. I did not change much of the code besides tweaking settings related to output directories and config settings. I also did not explore any advanced techniques beyond the initial LLM training such as fine-tuning or RAG.

Additionally, I have included the file main_job.sh which I used to submit a supercompute job on ASU's Sol supercomputer to train Caissa.

You can train your own model in reasonable time (or unreasonable time if you don't have a GPU) by running the commands in the .\train-caissa directory's README.

## Evaluation (.\evaluate-caissa)

## Results (.\results)


## Models

All Caissa v.0 checkpoints (500k, 1mil, 1.5mil iterations) are located in the directory .\caissa-v.0.

## TODOs

- Train Caissa v.0 for a larger number of iterations and/or optimize config file train_caissa.py
- Gather large dataset of position + best move pairs in the format FEN-Bestmove using Stockfish 17 at depth=20 (Here are a few scary words to warn you away from trying depth=25: multicore processing, hanging workers, race condition, deadlock, Gradescope-style debugging).
- Retrain model with identical config to Caissa v.0 and test ability to play full games. Currently, Caissa v.0 shows very good performance for chess puzzle-type positions which only capture two types of positions: 
    1. One side to move and win with significant advantage. 
    2. One side to move and draw ("saving" the game from a "lost" position).
- Train an even larger model on a larger/higher quality dataset -> Repeat indefinitely!!!

## acknowledgements

All code in the .\train-caissa directory was initially forked from Andrej Karpathy's nanoGPT repository.

[https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

The author acknowledges Research Computing at Arizona State University for providing computing and storage resources to train all checkpoints of the Caissa LLM.

Jennewein, Douglas M. et al. "The Sol Supercomputer at Arizona State University." In Practice and Experience in Advanced Research Computing (pp. 296–301). Association for Computing Machinery, 2023. 


