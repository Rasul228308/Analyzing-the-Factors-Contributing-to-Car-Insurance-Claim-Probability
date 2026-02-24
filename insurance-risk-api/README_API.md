## 🚀 Running the API

```bash
git clone https://github.com/Rasul228308/Analyzing-the-Factors-Contributing-to-Car-Insurance-Claim-Probability
cd insurance-risk-api
docker-compose up -d
```

Open [**http://localhost:8000/docs**](http://localhost:8000/docs) for the interactive API.

### Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"AGE": 0, "DRIVINGEXPERIENCE": 0, "CREDITSCORE": 0.2, 
       "SPEEDINGVIOLATIONS": 5, "DUIS": 2, "PASTACCIDENTS": 3, ...}'
```

### Example response
```json
{
  "claim_probability": 0.8732,
  "risk_tier": "Very High", 
  "prediction": 1
}
```
