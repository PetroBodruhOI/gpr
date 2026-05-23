# Testing GPR Worker

Test suite covers:
- Progress tracking (`_set_progress`)
- Task status retrieval (`get_task`)
- Soft voting aggregation (multi-chunk consensus)
- Error handling & validation
- User feedback submission
- Concurrent task isolation
- Edge cases (short/long audio, confidence bounds)

## Setup

```bash
pip install -r requirements-test.txt
```

## Run all tests

```bash
pytest
```

## Run specific test

```bash
pytest test_app.py::test_soft_voting_consensus -v
```

## Coverage report

```bash
pytest --cov=app --cov-report=html test_app.py
# Open htmlcov/index.html
```

## Run only unit tests (no external deps)

```bash
pytest -m unit
```

## Key test categories

### Progress tracking
- ✅ Initial pending state
- ✅ Progress updates overwrite previous
- ✅ Final result stored with prediction

### Task retrieval
- ✅ Missing task returns pending
- ✅ In-flight task shows progress
- ✅ Completed task returns full result

### Soft voting (multi-chunk ensemble)
- ✅ Single chunk: uses chunk prediction
- ✅ Multiple chunks (consensus): high confidence
- ✅ Multiple chunks (disagreement): ensemble shifts
- ✅ Uniform distribution: picks first by argmax

### Error handling
- ✅ Invalid feedback rating rejected (400)
- ✅ Feedback for nonexistent task rejected (404)
- ✅ Valid feedback stored with 30-day TTL

### Concurrent tasks
- ✅ Two tasks don't interfere
- ✅ Both tasks complete independently

### Edge cases
- ✅ Very short audio (< 1 chunk)
- ✅ Very long audio (10+ chunks)
- ✅ Confidence always in [0, 1]
