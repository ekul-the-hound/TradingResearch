# test_api_key.py
import config
from anthropic import Anthropic

print("="*70)
print("API KEY DIAGNOSTIC TEST")
print("="*70)

# Check if key exists
if not config.CLAUDE_API_KEY:
    print("[FAIL] No API key found in config!")
    exit()

print(f"[OK] API key loaded from file")
print(f"  Key starts with: {config.CLAUDE_API_KEY[:15]}...")
print(f"  Key ends with: ...{config.CLAUDE_API_KEY[-10:]}")
print(f"  Key length: {len(config.CLAUDE_API_KEY)} characters")
print(f"  Expected length: ~108 characters")

# Check for common issues
if ' ' in config.CLAUDE_API_KEY:
    print("[WARN]  WARNING: Key contains spaces!")
if '\n' in config.CLAUDE_API_KEY:
    print("[WARN]  WARNING: Key contains line breaks!")
if not config.CLAUDE_API_KEY.startswith('sk-ant-'):
    print("[WARN]  WARNING: Key doesn't start with 'sk-ant-'")

# Try to use it
print("\nTesting API connection...")
try:
    client = Anthropic(api_key=config.CLAUDE_API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say 'API test successful!' and nothing else."}]
    )
    print("[OK] SUCCESS! API key is valid!")
    print(f"Claude says: {message.content[0].text}")
except Exception as e:
    print(f"[FAIL] FAILED! Error: {e}")
    print("\nThis means your API key is invalid. You need to:")
    print("1. Go to https://console.anthropic.com/settings/keys")
    print("2. DELETE the current key")
    print("3. CREATE a new key")
    print("4. Copy it VERY carefully")
    print("5. Paste it into BacktestingAgent_API_KEY.txt")

print("="*70)