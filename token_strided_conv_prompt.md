
# Adding Token-strided convolutions

This is the modded-nanogpt repo, used for nanogpt speedrunning.

I would like to implement and Idea I call _token-strided convolutions_: the basic idea is that current tokenization with an embedding layer discards all of the potentially valuable information at the utf-8 byte level: for example, "The " and "the " are at initialization assigned completely independent embeddings to each other, further valuable information such as "does this token end in whitespace" and "does this token end in -ing" are discarded.

In the current implementation, the embedding is the sum of the current token embedding and the bigram embedding:

```python
# ---- Embeddings and input preparation ----
x = self.embed(input_seq) # embed is synced from lm_head during tied phase by optimizer

x0_bigram = self.bigram_embed(bigram_input_seq)[None]
 # Initialize residual stream with pre-layer-0 bigram injection
x = x + x0_bigram * bigram_lambdas[0]
```

To this, I would like to add the following:

```python
x_byte = self.byte_convolution(byte_seq)
```

We construct `byte_seq` as a tensor of size `(input_length, byte_window_size)`, where `byte_window_size = 8` is a new hyperparameter that we should sweep. In the abstract, byte_seq should be constructed using a by first tokenizing the input, and then ordering the utf-8 bytes associated with each token.

For example:

- Take the sequence "token strided convolutions are cool". (To help with parsing: the byte sequence here is `t o k e n _ s t r i d e d _ c o n v o l u t i o n s _ a r e _ c o o l`)
- This gets tokenized into "token" " str" "ided" " conv" "olutions" " are" " cool".
- This would then be put into the following shape in the `byte_seq` tensor:

|   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|
|   |   |   |   | o |   |   |
|   |   |   |   | l |   |   |
|   |   |   |   | u |   |   |
| t |   |   | _ | t |   | _ |
| o | _ | i | c | i | _ | c |
| k | s | d | o | o | a | o |
| e | t | e | n | n | r | o |
| n | r | d | v | s | e | l |

Whether we pad left or right is also something to ablate

- This would then have: An individual byte-level embedding layer applied to achieve a tensor which has a logical shape of `(input_length, byte_window_size, byte_embedding_dim)`, followed by a convolution/linear layer which will project this down to `(input_length, embedding_dim)`, the `self.byte_convolution` output. the output of  feel free to change the true tensor sizes to make it fast. 
