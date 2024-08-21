To update the combinatorial tests;

- open `\test\combinatorial\ACTS-model-validonly.xml` in ACTS

- modify the model & save the changes

- generate new combinations & export as `\test\combinatorial\combinations.csv`

- manually add the test for `iter > 1`: `*,*,*,...,*,2`

- add/change new dictionaries/values in `\test\combinatorial\input_values.jl` accordingly
