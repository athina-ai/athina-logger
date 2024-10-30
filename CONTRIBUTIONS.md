# installation
to install we need python3.9. Run 
```
make setup
```
to install python3.9 on mac. 

Then, run 
```
make build
```
to create the venv and install the requirements in it. 

now you can run the below command to activate your virtual environment. 
```bash
source venv/bin/activate 
```

# run unit tests locall

Make sure that your local backend api is running, or a staging environment. Otherwise, the api calls will go to production.

copy the `tests/.env.test` file into a `tests/.env`. Fill in the api you want to hit and your api key. 

run the tests with `make test` or `pytest`. Note that logs in tests do not go to stdout unless you run either 
`pytest -s` or `make test-debug`.
