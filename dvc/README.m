******* first ************************************
Commit source files to Git
$ git init
$ git add src
$ git commit -m "Initial commit"


Initialized DVC repository.
$ dvc init


Commit created dvc related files to Git
$ git add .dvc .dvcignore
$ git commit -m "Initialize DVC"


add diamonds.csv to DVC tracking:
$ dvc add data/diamonds.csv

Check dvc data file
$ cat data/diamonds.csv.dvc


$ git add data
$ git commit -m "Add initial version of the dataset to DVC"

*********second *************************************
We change the size of the dataset concatenating it to itself

Check changes
$ dvc status


We track changes
$ dvc add data/diamonds.csv

To capture the change to diamonds.csv.dvc, Git commit:
$ git add data/diamonds.csv.dvc
$ git commit -m "Add more rows to diamonds.csv"

***********Third***************************
Check git commit history
$ git log --oneline


Back to a Data version
git checkout 1f407bc

A good practice if create a new branch to ztrack the new changes
$ git switch -c new_branch

If yur rename the data in this branch, then you create a new version

To be safe, whenever you call git checkout, you should always call dvc checkout as well.

********** Create pipelines *******************

Add stages and pipeline dependencies
$ dvc stage add -n split \
               -d data/diamonds.csv -d src/split.py \
               -o data/train.csv -o data/test.csv \
               python src/split.py

Add training stages as well
$ dvc stage add -n train \
               -d data/train.csv -d data/test.csv -d src/train.py \
               -o models/model.joblib \
               -M metrics.json \ 
               python src/train.py

RUn the stages with (or editing dvc.yml)
$ dvc repro               

Commit changes
git commit -m "initial run of the pipeline"

$ git push; dvc push