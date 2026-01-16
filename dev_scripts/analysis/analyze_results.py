#!/usr/bin/env python
"""Analyze ACE vs ThatOtherContextEngine benchmark results."""
import json

with open('benchmark_results/ace_ThatOtherContextEngine_1000_20260106_122712.json') as f:
    data = json.load(f)

print('=' * 80)
print('BENCHMARK ANALYSIS - ACE vs ThatOtherContextEngine (250 queries completed before timeout)')
print('=' * 80)
print()
print('SUMMARY STATS:')
print(f'  Total Queries: {data["total_queries"]}')
print(f'  ACE Wins: {data["stats"]["ace_wins"]} ({data["stats"]["ace_win_rate"]}%)')
print(f'  ThatOtherContextEngine Wins: {data["stats"]["ThatOtherContextEngine_wins"]} ({data["stats"]["ThatOtherContextEngine_win_rate"]}%)')
print(f'  Ties: {data["stats"]["ties"]} ({data["stats"]["tie_rate"]}%)')
print()
print('BY CATEGORY:')
for cat, stats in data['stats']['by_category'].items():
    total = stats['ace'] + stats['ThatOtherContextEngine'] + stats['tie']
    ace_pct = (stats['ace'] / total * 100) if total > 0 else 0
    status = "PERFECT" if stats['ThatOtherContextEngine'] == 0 else f"NEEDS WORK (-{stats['ThatOtherContextEngine']})"
    print(f'  {cat:25} ACE: {stats["ace"]:2} | ThatOtherContextEngine: {stats["ThatOtherContextEngine"]:2} | {status}')

print()
print('=' * 80)
print('ALL 29 ThatOtherContextEngine WINS - ROOT CAUSE ANALYSIS')
print('=' * 80)

ThatOtherContextEngine_wins = [r for r in data['results'] if r['winner'] == 'ThatOtherContextEngine']

# Group by pattern
test_file_issue = []
json_file_issue = []
other_issue = []

for r in ThatOtherContextEngine_wins:
    ace1 = r['ace_files'][0]
    if 'test' in ace1.lower() or ace1.startswith('tests/'):
        test_file_issue.append(r)
    elif ace1.endswith('.json'):
        json_file_issue.append(r)
    else:
        other_issue.append(r)

print(f'\n1. TEST FILE RANKED #1 ({len(test_file_issue)} cases)')
print('-' * 60)
for r in test_file_issue:
    exp = r['expected_files'][0] if r['expected_files'] else 'N/A'
    ace1 = r['ace_files'][0]
    print(f'  Query: {r["query"][:50]}')
    print(f'    Expected: {exp}')
    print(f'    ACE #1:   {ace1}')
    print()

print(f'\n2. JSON FILE RANKED #1 ({len(json_file_issue)} cases)')
print('-' * 60)
for r in json_file_issue:
    exp = r['expected_files'][0] if r['expected_files'] else 'N/A'
    ace1 = r['ace_files'][0]
    print(f'  Query: {r["query"][:50]}')
    print(f'    Expected: {exp}')
    print(f'    ACE #1:   {ace1}')
    print()

print(f'\n3. OTHER ISSUES ({len(other_issue)} cases)')
print('-' * 60)
for r in other_issue:
    exp = r['expected_files'][0] if r['expected_files'] else 'N/A'
    ace1 = r['ace_files'][0]
    print(f'  Query: {r["query"][:50]}')
    print(f'    Expected: {exp}')
    print(f'    ACE #1:   {ace1}')
    print()

print('=' * 80)
print('ROOT CAUSE SUMMARY')
print('=' * 80)
print(f'  Test files outranking source: {len(test_file_issue)} ({len(test_file_issue)/29*100:.1f}%)')
print(f'  JSON files outranking source: {len(json_file_issue)} ({len(json_file_issue)/29*100:.1f}%)')
print(f'  Other issues:                 {len(other_issue)} ({len(other_issue)/29*100:.1f}%)')
print()
print('CONCLUSION: Fix test file penalty and JSON file penalty to achieve 100% ACE superiority')
