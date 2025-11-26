# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

init_from = 'scratch'
out_dir = 'out-caissa'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'caissa-puzzles-12m'
wandb_run_name = 'caissa-puzzles-12m'

dataset = 'chess-data'
gradient_accumulation_steps = 4
batch_size = 512
block_size = 128 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.05

learning_rate = 6e-4 # with baby networks can afford to go a bit higher
max_iters = 10000 # TODO: change max_iters and lr_decay_iters to an actual large number
lr_decay_iters = 10000 # make equal to max_iters usually
min_lr = 6e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 2000 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
