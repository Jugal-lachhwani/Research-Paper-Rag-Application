import json
import os

# Path to the directory containing your original full datasets
base_dir = "/home/jugal/Documents/DataScience/Research-Paper-Rag-Application/open_ragbench/pdf/arxiv"
# The directory where you want to save the filtered output
out_dir = "/home/jugal/Documents/DataScience/Research-Paper-Rag-Application"

pdf_urls_path = os.path.join(base_dir, "pdf_urls.json")
qrels_path = os.path.join(base_dir, "qrels.json")
queries_path = os.path.join(base_dir, "queries.json")

print("Loading data...")

# 1. Load PDF URLs and get the first 100
with open(pdf_urls_path, "r") as f:
    pdf_urls = json.load(f)
    
first_100_doc_ids = list(pdf_urls.keys())[:100]
first_100_doc_ids_set = set(first_100_doc_ids)

first_100_urls = {k: pdf_urls[k] for k in first_100_doc_ids}

# 2. Filter qrels to only include the first 100 documents
with open(qrels_path, "r") as f:
    qrels = json.load(f)

qrels_100 = {}
for q_id, q_data in qrels.items():
    if q_data["doc_id"] in first_100_doc_ids_set:
        qrels_100[q_id] = q_data
        
# 3. Filter queries to only include queries that are present in the filtered qrels
with open(queries_path, "r") as f:
    queries = json.load(f)

queries_100 = {}
for q_id in qrels_100.keys():
    queries_100[q_id] = queries[q_id]

# 4. Save to new JSON files
print("Saving filtered datasets...")
with open(os.path.join(out_dir, "pdf_urls_100.json"), "w") as f:
    json.dump(first_100_urls, f, indent=4)
    
with open(os.path.join(out_dir, "qrels_100.json"), "w") as f:
    json.dump(qrels_100, f, indent=4)
    
with open(os.path.join(out_dir, "queries_100.json"), "w") as f:
    json.dump(queries_100, f, indent=4)

# 5. Print the counts
print("\n" + "="*40)
print("SUCCESS: Data filtering complete!")
print("="*40)
print(f"Total documents targeted:  {len(first_100_urls)}")
print(f"Total 'qrels' found:       {len(qrels_100)}")
print(f"Total 'queries' found:     {len(queries_100)}")
print("="*40)
print("Saved files to:")
print(f"- {os.path.join(out_dir, 'pdf_urls_100.json')}")
print(f"- {os.path.join(out_dir, 'qrels_100.json')}")
print(f"- {os.path.join(out_dir, 'queries_100.json')}")
