Due to official training set is only available to task participants, you cannot reproduce the training set and models used in this repository. However, official test set is available to public [here](https://zenodo.org/record/6362498#.YnMxb_PML0o). I already put it in data folder. The model can be downloaded from [here](https://arizona.box.com/s/mqilthshz1x66z87ljzv521uesak4x59). Make sure the model's folder is in the output folder as in "output/distilbert-base-uncased". Here is the steps to reproduce the scores of the performance on the official test set.
- on the command line, go to repo, type "docker build - < Dockerfile"
- type "docker images" to get the tag of the image from the table 
- type "docker tag _tag_ shared-task-bulut" (replace _tag_ with tag you got) 
- run the image by typing "docker run -it -p 7777:9999 -v "$PWD:/app/" shared-task-bulut:latest"
- go to "app/scripts" directory, run "python3 create_test_set.py"
- check data/preprocessed to see the file "test.csv"
- run "python3 evaluate_on_test.py"
- the scores will be printed out along with submission file in "submissions" folder
