1 - poetry install - pip install poetry if u dont have it
2 - start the ngrok server - ngrok http 8000
3 - run the setup models program to select the model
4 - paste the ngrok server url in .env
5 - customize the arangoDB details in .env
6 - start arangoDB in docker
7 - poetry run python existing_repo_workflow.py
8 - configure the program to work with your repos (commands below)
9 - create a webhook in ur code repo - the end point and secret should be gotten from configuring the program (step above)
10 - change the content type of the webhook to application/json and create webhook 
11 - save ur documentation embdeddings in arangoDB if they are not there already (command below) with the config id given in step 8
12 - u can now just commit anything and the program should work

** in the env u only need to change arangoDB configs, ngrok url, and 1 of the api keys if u are using them
** make sure ur documentation is in a folder named docs

CONFIGURE COMMANDS

gitlab - gitlab:
Invoke-WebRequest -Uri "http://localhost:8000/configure" -Method POST -ContentType "application/json" -Body '{ "code_provider": "gitlab", "docs_provider": "gitlab", "gitlab_code_repo_url": "", "gitlab_code_project_id": "", "gitlab_code_token": "", "gitlab_docs_repo_url": "", "gitlab_docs_project_id": "", "gitlab_docs_token": "", "docs_folder": "docs"}' | Select-Object -ExpandProperty Content

Github - github:
Invoke-WebRequest -Uri "http://localhost:8000/configure" -Method POST -ContentType "application/json" -Body '{ "code_provider": "github", "docs_provider": "github", "github_repo_url": "", "github_token": "", "github_docs_repo_url": "", "github_docs_token": "", "docs_folder": "docs"}' | Select-Object -ExpandProperty Content

Github-Gitlab:
Invoke-WebRequest -Uri "http://localhost:8000/configure" -Method POST -ContentType "application/json" -Body '{ "code_provider": "github", "docs_provider": "gitlab", "github_repo_url": "", "github_token": "", "gitlab_docs_repo_url": "", "gitlab_docs_token": "", "gitlab_docs_project_id": "", "docs_folder": "docs"}' | Select-Object -ExpandProperty Content

Gitlab - Github:
Invoke-WebRequest -Uri "http://localhost:8000/configure" -Method POST -ContentType "application/json" -Body '{ "code_provider": "gitlab", "docs_provider": "github", "gitlab_code_repo_url": "", "gitlab_code_project_id": "", "gitlab_code_token": "", "github_docs_repo_url": "", "github_docs_token": "", "docs_folder": "docs"}' | Select-Object -ExpandProperty Content


EMBEDDINGS COMMAND

Invoke-WebRequest -Uri "http://localhost:8000/generate-embeddings/replace-with-config-here" -Method POST -ContentType "application/json" | Select-Object -ExpandProperty Content