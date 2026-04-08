from fastapi import FastAPI
from environment import EmailTriageEnv
import uvicorn

app = FastAPI(title='OpenEnv Email Triage')

@app.get('/')
def root():
    return {'name': 'email-triage-env', 'status': 'ready', 'tasks': [1, 2, 3]}

@app.get('/health')
def health():
    env = EmailTriageEnv()
    obs = env.reset(task_id=1)
    return {'status': 'ok', 'inbox_size': len(obs.inbox)}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=7860) # HF Spaces default port