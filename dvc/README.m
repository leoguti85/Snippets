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