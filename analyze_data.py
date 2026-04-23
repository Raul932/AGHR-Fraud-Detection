import json, pandas as pd, sys, os
sys.stdout.reconfigure(encoding='utf-8')

data_dir = 'The Truman Show - train'
df = pd.read_csv(f'{data_dir}/transactions.csv')
print('COLUMNS:', list(df.columns))
print(f'Total: {len(df)} transactions')
print()

# All transactions detail
for _, row in df.iterrows():
    desc = str(row.get('description', ''))
    tid = row['transaction_id']
    amt = row['amount']
    ts = row['timestamp']
    method = str(row.get('payment_method', ''))
    loc = str(row.get('location', ''))
    sender = str(row.get('sender_id', ''))
    txtype = str(row.get('transaction_type', ''))
    bal = row.get('balance_after', 0)
    is_rent = 'Rent' in desc or 'rent' in desc
    marker = '  ' if is_rent else '>>'
    print(f'{marker} [{sender[:20]}] {tid[:8]}.. | {desc[:50]:50s} | {amt:10.2f} | {txtype:20s} | {method:15s} | {loc[:30]:30s} | {ts} | bal={bal}')

print()
# Load phishing
sms = json.loads(open(f'{data_dir}/sms.json').read())
mails = json.loads(open(f'{data_dir}/mails.json').read())
print(f'SMS count: {len(sms)}, Mail count: {len(mails)}')
print()
print('=== PHISHING SMS SNIPPETS ===')
for s in sms:
    txt = s['sms'].replace('\n', ' ')[:200]
    if any(w in txt.lower() for w in ['paypa', 'amaz0n', 'ub3r', 'netfl1x', 'deutschebank', 'r1d3share', 'verify', 'urgent', 'suspicious']):
        print(f'  PHISH: {txt}')
print()
print('=== PHISHING MAIL SNIPPETS ===')
for m in mails:
    txt = m['mail'].replace('\n', ' ')[:200]
    if any(w in txt.lower() for w in ['paypa', 'amaz0n', 'ub3r', 'netfl1x', 'deutschebank', 'r1d3share', 'verify', 'urgent', 'suspicious']):
        print(f'  PHISH: {txt}')
