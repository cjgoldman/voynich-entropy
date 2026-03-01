Use Entropy Model from FAIR BLT work to measure the next byte entropy of Voynich text. 
# Background

## Next Byte Entropy
### Definition & Calculation
The next byte entropy is defined as:

$$
H\left(x_i\right)=-\sum_{v \in \mathcal{V}} p_e\left(x_i=v \mid \boldsymbol{x}_{<i}\right) \log p_e\left(x_i=v \mid \boldsymbol{x}_{<i}\right)
$$

The entropy model is used to estimate the local conditional entropy using the estimated probabilities for each byte (character in our case). Computing the estimated entropy is done by the `entropy()` function in `patcher.py` 

``` python
def entropy(scores):
    # scores: [bs, seq_len, vocab=260]
    log_probs = F.log_softmax(scores, dim=-1)   # log p(byte | context)
    probs = torch.exp(log_probs)                  # p(byte | context)
    p_log_p = log_probs * probs                   # p * log(p)
    entropy = -p_log_p.sum(dim=-1)                # H = -Σ p log p
    return entropy  # shape: [bs, seq_len]
```


### Entropy Boundary Detection
The paper uses two methods to detect patch boundaries. First is a simple global threshold. The second is an approximate monotonicity method that can be interpreted as identifying points that break approximate monotonically decreasing entropy within the patch.
# Phase 1: Set Up and Simple Test
The goal of the first phase is to modify the pre-trained Entropy Model from FAIR's BLT work to process Voynich and other non-ascii characters. I think there are two approaches:
1. Voynich -> UTF-8 and use model as is. This is probably the right way to go.
2. Add Voynich vocab tokens to the model. This will require a bit of surgery on the model. Probably should try this if the first attempt doesn't work. 
## Voynich -> UTF-8 & Use Model As Is:
The thing to remember is that the BLT model is a _byte_ model more than a _character_ model. Where a regular LLM would have tens of thousands of tokens (one for each sub word), this model only has 256 + 4 tokens (one for each value of a byte and an additional four control tokens). So when the model is fed a character it first breaks it down into bytes according to the [UTF-8 standard.](https://en.wikipedia.org/wiki/UTF-8) These bytes are then fed into the model. The key takeaway is: _the model takes in single bytes for some characters and multiple bytes for others depending on the UTF-8 encoding._

### Understanding UTF-8   
Unfortunately, we sort of have to understand the basics of UTF-8 encoding scheme to understand how the model should handle the novel Voynich characters. So here we go ...

The scheme begins by assigning every character an index number. The index number is called a _code point_ and written as a `U+HEX` code, where the hex code is typically 4 to 6 digits long. There are 1,112,064 valid code points in the unicode standard.  ASCII gets 1 through 128 with the remaining latin based characters and most non East Asian character encoded between 129 and 2048. Most East Asian characters are given one of the next 61,440 code points. The remaining 1,048,576 code points are used for emojis, and everything else. The Private Use Area is contained between U+E000 and U+F8FF. 

**How UTF-8 distributes code point bits into bytes:**
The UTF-8 scheme _dynamically_ increases the number of bytes used based on the numerical value of the code point. This can range from 1 to 4 bytes. Each byte is divided into prefix bits and payload bits. The arrangement of the prefix and payload bits is determined by the following rules:  
-  **Byte 1 prefix:** The prefix of the first byte in the code is a set of leading `1`s equals the total byte count followed by a `0` (`110` = 2 bytes, `1110` = 3, `11110` = 4). Single-byte codes start with `0`.
- **Continuation bytes:** Always prefixed with `10`, making them distinguishable from leading bytes.
- **Payload (`x`) slots:** Filled right-to-left with the code point's binary value, zero-padded on the left as needed.
   
For the following table, assume that a code index for a character is given by a 4 digit hex code. U+> $\textcolor{#BD8DE2}{\text{X}}$ $\textcolor{#FF0000}{\text{X}}$ $\textcolor{#00B050}{\text{X}}$ $\textcolor{#0070C0}{\text{X}}$ . This hex code can be written in binary as 4 quartets: 
binary = $\textcolor{#BD8DE2}{\text{xxxx}}$ $\textcolor{#FF0000}{\text{xxxx}}$  $\textcolor{#00B050}{\text{xxxx}}$  $\textcolor{#0070C0}{\text{xxxx}}$

| Bytes | Code Point       | Byte 1                                                                                          | Byte 2                                                                                          | Byte 3                                                                                          |
| ----- | ---------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 1     | U+0000 –– U+007F | 0 $\textcolor{#00B050}{\text{xxx}}$ $\textcolor{#0070C0}{\text{xxxx}}$ |                                                                                                 |                                                                                                 |
| 2     | U+0080 –– U+07FF | 110 $\textcolor{#FF0000}{\text{xxx}}$ $\textcolor{#00B050}{\text{xx}}$   | 10 $\textcolor{#00B050}{\text{xx}}$ $\textcolor{#0070C0}{\text{xxxx}}$ |                                                                                                 |
| 3     | U+0800 –– U+FFFF | 110 $\textcolor{#BD8DE2}{\text{xxxxx}}$                                          | 10 $\textcolor{#FF0000}{\text{xxxx}}$ $\textcolor{#00B050}{\text{xx}}$   | 10 $\textcolor{#00B050}{\text{xx}}$ $\textcolor{#0070C0}{\text{xxxx}}$ |


**Worked example — `é` = `U+00E9`:**
Decimal: 233   
Binary:  $\textcolor{#BD8DE2}{\text{0000}}$ $\textcolor{#FF0000}{\text{0000}}$ $\textcolor{#00B050}{\text{1110}}$ $\textcolor{#0070C0}{\text{1001}}$

| Code Point | Byte 1                                                                                        | Byte 2                                                                                          |
| ---------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| U+00E9     | 110 $\textcolor{#FF0000}{\text{000}}$ $\textcolor{#00B050}{\text{11}}$ | 10 $\textcolor{#00B050}{\text{10}}$ $\textcolor{#0070C0}{\text{1001}}$ |

### How To Incorporate Voynich
The basic plan would be to assign UTF-8 code points to the existing transcription characters from the voynich-attack repo. Assuming we hav $N_V$ characters we could just use the first $N_V$ code points after U+E000. Alternatively, we could match the characters from the voynich-attack repo to the closest subset of [EVA Unicode](https://www.kreativekorp.com/software/fonts/voynich/) font. The only reason for doing this is for the convenience of being able to display characters rather than raw UTF-8 when we are looking at the entropy charts.   

Once we have the encoding mapped we just need to create JSON files with the UTF-8 mapped Voynich. The model uses a basic JSON format for the input.

Example of Standard
   ``` json
{"sample_id": "0", "text": "The quick brown fox..."}
{"sample_id": "1", "text": "Another document here..."}
   ```

It should be a be to handle the Voynich UTF-8. If the local system doesnt have the fonts stored, it should default to `\u` format.

Example of Voynich
   ``` json
{"sample_id": "0", "text": "This is regular text..."}
{"sample_id": "1", "text": "\uE000\uE001\uE002\uE003..."}
   ```

This should allow us to use the models existing data pipeline without any major modifications.

### First Runs
#### Baseline
The first run should be pretty straight forward, just need to run the Voynich text through the pre-trained model to get a baseline. This should show a fairly high and uniform level of entropy. This will also act as a basic "smoke test" to make sure the basic setup is working. 

#### Fine-Tuning 
We should initially try two different fine tuning techniques and see what we get.
1. The first round of fine-tuning should be a simple train/test split where we only use Voynich data to fine-tune. 
2. Second round should be a data mixing fine-tuning where we mix in a certain amount of the original training data to try to prevent catastrophic forgetting. 
## Extending the Token Space
We probably don't want to do this, but I'm putting it in for completeness. The only reason that I think that this would work is if the Voynich characters behave very differently than normal characters or glyphs yet some how combine to follow a language like grammar. 

This will require two modifications. 
1. **Expanded Character Vocab:** The pre-trained model has a vocabulary of 260 (256 byte tokens+ 4 control tokens). Assume we want to use $N_V$ new Voynich characters - We will have to augment the existing embedding matrix and output projection matrix, currently $[260,768]$ and $[768,260]$ respectively. The new matrices will have to be  $[260+N_V,768]$ and $[768,260+N_V]$, where the new entries will be initialized with the mean value of the original in order to minimize the impact on the existing characters. 
2. **Modify the Input Pipeline:** The data format for the original work is a simple JSON file. There is some preprocessing to shuffle samples etc but the big picture is that the text strings from json file get processed by `BltTokenizer()` found in `bytelatent/tokenizers/blt_tokenizer.py` . The `BltTokenizer()` expects that all the text is UTF-8. We can probably use the same approach to putting the Voynich data into JSON as used above, but we will have to make a fair number of changes to the tokenizer logic. 
We will also have to be careful about training up the new token embeddings (eg freezing the bulk weights and using a very conservative learning rate).
