## Composite framework for capturing different indicators of (un)productive behavior in text

**Requires**:
* R installation
* spacy download "en_core_web_sm"
* Make sure to run ```setup_r.py``` before using politeness evaluator
* You need to set up a Google Cloud app w/ PerspectiveAPI enabled, get an API key, and set up a ```.env.local``` file with ```PERSPECTIVE_API_KEY=xxx``` **if you want to use the toxicity evaluator**
* You will need to grab HateBERT_hateval zip from the project Google Drive (too big to commit to GitHub)
