To launch the program, you must first visit thesrcdirectory, which can befound using the file directory schema listed earlier. Once you have a command prompt or terminal window open run the main.p yfile using the following arguments:

$: python main.py <path to model, i.e. .\saved_model, (optional)>

If a path to a saved model is provided, and it must be the parent directory called:
.\saved_model

The program will load a previously created model for testing. Otherwise,if no argument is provided then the program will attempt to build a model off of 30 epochs, and 10 songs for training and 10 songs for testin
