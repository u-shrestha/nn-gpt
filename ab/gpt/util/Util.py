import os.path
import re

# todo: Verify that the model's accuracy does not decrease by more than 10%, or increase at some epochs
def nn_accepted(nn_dir):
    accepted = True
    return accepted


# todo: Verify if model has implementation of all required methods, and use all mentioned hyperparameters, like 'lr', 'momentum'
# todo: Optimize code with library like 'deadcode' (after: pip install deadcode)
def verify_nn_code(nn_dir, nn_file):
    verified = True
    error_message = ''
    if not verified:
        with open(nn_dir / f"error_code_verification.txt", "w+") as error_file:
            error_file.write(f"Code verification failed: {error_message}")
    return verified

def exists(f):
    return f and os.path.exists(f)


def extract_str(s: str, pref: str, suf: str):
    if s.count(pref) > 0 and s.count(suf) > 0:
        x = re.search(f"{pref}((.|\s)*?){suf}", s)
        if x:
            res = x.group()
            res = res.replace("```python", "")
            res = res.replace("```", "")
            return res
    return None
