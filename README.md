
# Caissa (nanoGPT fork)

Chess-based LLM trained exclusively on position + best move pairs without game context. 
Aims to explore the limits of natural-language-based processing for chess tasks along with searchless evaluation.  

## Installation and Dependencies

For .\train-caissa and .\caissa-v.0
```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

For .\evaluate-caissa and .\create-train-val-puzzle-sets
```
pip install torch tiktoken openai python-dotenv chess pickle json matplotlib numpy
```

You will also need an OpenAI API key in your .env file to test the different GPT models on the validation puzzles test suite.

## Subdirectory READMEs

I have included another README file in each subdirectory with more details. This top-level README provides a top-level  explanation (who could have known?) of what this repository actually is and each folder's overarching purpose.

## Dataset (.\create-train-val-puzzle-sets)

I gathered the dataset for training Caissa from [Lichess's open source puzzle database](https://database.lichess.org/#puzzles).
The provided files parse the CSV data into a single text file for training data, along with holding out a sample for validation. Additional functionality includes filtering all validation puzzle positions from the initial training data along with counting the number of positions + puzzles.

## Training (.\train-caissa)

All code in the .\train-caissa directory (which was used to train Caissa [duh!]) was initially forked from Andrej Karpathy's nanoGPT repository. I did not change much of the code besides tweaking settings related to output directories and config settings. I also did not explore any advanced techniques beyond the initial LLM training such as fine-tuning or RAG.

Additionally, I have included the file main_job.sh which I used to submit a job on ASU's Sol supercomputer for training Caissa.

You can train your own model in reasonable time (or unreasonable time if you don't have a GPU) by running the commands in the .\train-caissa directory's README.

## Evaluation (.\evaluate-caissa)

I performed two sets of tests for overall model accuracy/sanity along with model speed (response time per puzzle).

### Test #1: Evaluate overall model accuracy and sanity on a large set of chess puzzles.
All files located in the .\evaluate-caissa\detailed-tests subdirectory.
Tested GPT-5 (minimal), GPT-5-mini (minimal), GPT-5-nano (minimal), GPT-3.5-turbo, Stockfish 17 (thinktime=1s), Caissa-v0-iters-500k, Caissa-v0-iters-1m, Caissa-v0-iters-1.5m on a comprehensive test suite of n=1000 puzzles. 
Themes included: mate_in_1, mate_in_2, mate_in_3, one_move, middlegame, endgame, zugzwang (also known as zuggie in some circles), crushing, master_vs_master, superGM 

### Test #2: Evaluate model speed (response times) for a small sample of chess puzzles.
All files located in the .\evaluate-caissa\speed-tests subdirectory.
Tested GPT-5 (medium/minimal), GPT-5-mini (minimal), GPT-5-nano (minimal), Stockfish 17 (thinktime=0.01s,0.05s,0.10s,0.15s), Caissa-v0-iters-1.5m on a small test suite of n=10 superGM theme puzzles to measure response times.

## Results (.\results)

This folder contains graphics (in .\graphics) of Test #1 and #2 performed during Evaluation. It also includes the raw JSON files (in .\JSONs) with each model's results in case someone wishes to replicate my tests.

## Models (.\caissa-v.0)

A Google drive link is included in the README for .\caissa-v.0, including all Caissa v.0 checkpoints (500k, 1mil, 1.5mil iterations) and its vocabulary file.

## TODOs

- Train Caissa v.0 for a larger number of iterations and/or optimize config file train_caissa.py
- Gather large dataset of position + best move pairs in the format FEN-Bestmove using Stockfish 17 at depth=20 (Here are a few scary words to warn you away from trying depth=25: multicore processing, hanging workers, race condition, deadlock, Gradescope-style debugging).
- Retrain model with identical config to Caissa v.0 and test ability to play full games. Currently, Caissa v.0 shows very good performance for chess puzzle-type positions which only capture two types of positions: 
    1. One side to move and win with significant advantage. 
    2. One side to move and draw ("saving" the game from a "lost" position).
- Train an even larger model on a larger/higher quality dataset -> Repeat indefinitely!!!

## Acknowledgements

All code in the .\train-caissa directory was initially forked from Andrej Karpathy's nanoGPT repository.

[https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

The author acknowledges Research Computing at Arizona State University for providing computing and storage resources to train all checkpoints of the Caissa LLM.

Jennewein, Douglas M. et al. "The Sol Supercomputer at Arizona State University." In Practice and Experience in Advanced Research Computing (pp. 296–301). Association for Computing Machinery, 2023. 


