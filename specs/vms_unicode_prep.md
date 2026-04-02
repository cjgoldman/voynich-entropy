# Purpose
This document describes the process of preparing voynich unicode data for ingest into the blt entropy model(a byte-level transformer model).
# Initial Data Format 
The voynich unicode is available in a dataframe format.
The columns are as follows:
- `folio`: The folio and side of the manuscript page (e.g., "1r" for folio 1 recto and "1v" for folio 1 verso).
- `par`: The paragraph number on the page.
- `line`: The line number within the paragraph.
- `t1`: The first "word" or "token" in the line.
    - Typically these are sequences of glyphs that are separated by commas but can occasionally be a single glyph.
    - Important: these are not tokens in the sense of a natural language tokenizer, but rather sequences of glyphs that are treated as units for analysis.
- `t2`: The second "word" or "token" in the line.
- `t3`: The third "word" or "token" in the line.

The columns go to `t26`, but many lines have fewer than 26 tokens, an empty cell is indicated by `$` in the dataframe.

# Specified Capabilities
The VMS unicode data preparation module should provide the following capabilities when preparing the data for the blt entropy model:
- **Byte Length**: User can specify the max byte length for the input sequences. The module should ensure that the prepared data does not exceed this byte length. and should not truncate bytes in the middle of a line. The module should also ensure that the byte length is calculated correctly, taking into account the encoding of the unicode characters. 
    - The default byte length should be set to 8192 bytes. 
    - If the prepared data exceeds the specified byte length, the module should raise an error or warning to alert the user that the data cannot be processed within the given constraints.
- **Comma Removal**: The module should remove commas from the tokens in the `t1` to `t26` columns, as commas are used as separators in the original data but are not needed for the blt model input.
- **Space Separation**: The module should concatenate the tokens from `t1` to `t26` into a single string for each line, with spaces separating the tokens. This will create a single input sequence for each line that can be fed into the blt model.
- **Empty Token Handling**: The module should handle empty tokens (indicated by `$`) appropriately by ignoring them in the concatenation process. This means that if a token is empty, it should not contribute to the final input string for that line.
- **Beginning of Paragraph Marker**: The module should prepend a pilcrow symbol (`¶`) to the beginning of the first line of each paragraph.
- **End of Paragraph Marker**: The module should append a paragraph separator symbol U+2029 (`\u2029`) to the end of the last line of each paragraph.
- **End of Line Marker**: The module should append a line separator symbol U+2028 (`\u2028`) to the end of each line. The last line of a paragraph should have a line separator followed by a paragraph separator to indicate the end of the paragraph.
- **Input Parameter**: The module should accept a dictionary specifying the start and end of the data to be prepared. The format of the dictionary should be as follows:
    ```python
    {
        "start": {
            "folio": "1r",
            "par": 1,
            "line": 1
        },
        "end": {
            "folio": "10v",
            "par": 5,
            "line": 10
        }
    }
    ```
    This allows the user to specify a range of data to be prepared, which can be useful for processing large datasets in batches or for focusing on specific sections of the manuscript. The module should validate the input parameters to ensure that they correspond to valid folio, paragraph, and line numbers in the dataset.
- **Output Format**: The module should output the prepared data in a list of strings format, where each string is a concatenated line of tokens with the appropriate markers for paragraphs and lines. 
