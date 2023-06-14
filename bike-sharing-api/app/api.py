from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

app = FastAPI()



@app.get("/predictbikeshare/")
async def predict_bike_share(param1: str='yes'):
    print("Entered",param1)
    #backend = BikeSharePredictor()
    result="Entered param "+param1
    
    #result=backend.make_prediction(data_in)
   
    print("Exiting")
    return {"result":result}
    
