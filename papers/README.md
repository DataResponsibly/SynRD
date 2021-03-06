# Overview
This file will serve as a development guide, providing instructions both for running existing publications/code and for adding new publications.

Please make sure to run `pip install -r requirements` in this directory from your conda environment, shell, or whatever development console you use, before trying to run our notebooks.

# Existing files
Broadly, this folder is structured like this: 
```
papers/
├─ paper1/
├─ paper2/
├─ paper3.../
│  ├─ pdf/
│  │  ├─ pdf.pdf
│  ├─ data (ignored)/
│  ├─ process.ipynb (ignore, messy)
│  ├─ paper3.py (publication class, clean)
├─ domains (for mst)/
│  ├─ paper1-domain.json...
├─ private_data/
│  ├─ paper_i/
│  │  ├─ e^?/
│  │  │  ├─ synthesizer/
│  │  │  │  ├─ iteration_i.pickle
├─ meta_classes.py
├─ useful_notebooks.ipynb
```
## Important Notes
### - Each "paper" here is named according to bibtex convention (authorYEARfirstword).

### - The data folders within each paper are gitignored to avoid data clutter.

### - The private_data/ folder contains only auto-generated private dataframe pickles for each paper. Folder architecture is self explanatory.

### - All abstract classes are in meta_classes.py

### - All data privatization and results can be regenerated by running the relevant .ipynb notebooks

## How to add a new paper
Brief details on how to add a new paper.

1. Create a new folder with (authorYEARfirstword)
2. Create a `process.ipynb` notebook as your data playground. Use this to investigate data cleaning/processing/results generation.
3. In parellel with (2), create a `authorYEARfirstword.py` file, and extend the `Publication()` metaclass with `AuthorYEARFirstword(Publication)`. Add the relevant details (see `meta_classes.py` for notes on what this means). Then, begin to move over `findings` from `process.ipynb` into replicable lambdas in `AuthorYEARFirstword(Publication)`.
4. Ensure that `AuthorYEARFirstword(Publication)` has a `FINDINGS` list class attribute. This should consist of `Finding` objects that wrap each `finding_i(self)` lambda in the proper `Finding, VisualFinding or FigureFinding` metaclass, and adds it to the list. 
5. See `Saw2018Cross` for an example of this a correctly implemented `Publication` class.

## Addendum on finding lambdas
`Finding` lambdas should have a particular structure that should be strictly adhered to. Consider the following example, and note particularly the return values
```Python
def finding_i_j(self): # there can be kwargs
    """
    (Text from paper, usually 2 or 3 sentences)
    """
    # often can use a table finding directly or 
    # as a starting point to quickly recreate 
    # finding
    results = self.table() 

    # (pandas stuff happens here to generate 
    # the findings)

    return ([values], 
            soft_finding, 
            [hard_findings])
```
The finding lambdas can essentially perform any computation necessary, but must return a tuple of
1. A list of values (these are a set of any relevant values to the soft finding, non-exhaustive)

    #### For example:
    ```Python
    [interest_stem_ninth,interest_stem_eleventh]
    ```

2. A soft_finding boolean (this is simply a boolean that reflects the primary inequality/contrast presented in the original paper for this finding)
    #### For example:
    ```Python
    soft_finding = interest_stem_ninth > interest_stem_eleventh
    ```

3. A list of hard findings i.e. values (this could be the difference or set of differences that affected the soft_finding inequality. F)
    #### For example:
    ```Python
    hard_finding = interest_stem_ninth - interest_stem_eleventh
    hard_findings = [hard_finding] 
    ```