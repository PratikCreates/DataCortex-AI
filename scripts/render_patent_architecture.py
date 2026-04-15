import os
import sys

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from graphviz import Digraph
except ImportError:
    print("Graphviz python library not found. Please run: pip install graphviz")
    sys.exit(1)

OUTPUT_DIR = os.path.join(ROOT, "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create sophisticated multi-agent architecture
dot = Digraph('DataCortexAI', comment='DataCortex-AI SOTA Architecture')
dot.attr(rankdir='TB', splines='ortho', bgcolor='#FAFBFC', 
         nodesep='0.8', ranksep='0.8', fontname='Helvetica', pad='0.4',
         dpi='300', compound='true')

# Refined styles
dot.attr('node', shape='box', style='filled,rounded', fontname='Helvetica',
         fontsize='10', margin='0.3,0.2', penwidth='2')
dot.attr('edge', fontname='Helvetica', fontsize='9', penwidth='2', 
         arrowsize='0.7')

# Title
dot.node('title', 'DataCortex-AI\nAutonomous Meta-Learning System', 
         shape='plaintext', fontsize='16', fontcolor='#1A202C', 
         fontname='Helvetica-Bold')

# User Node
dot.node('User', '👤\nData Scientist\n(User Intent)', shape='ellipse', fillcolor='#667EEA', 
         color='#5A67D8', fontcolor='white', fontsize='11', 
         width='1.2', height='1.2', penwidth='2.5', style='filled')

# The Planning Region (Kimi K2)
with dot.subgraph(name='cluster_planning') as c:
    c.attr(label='Meta-Cognitive Planning (Kimi-K2-Thinking)', 
           style='filled,rounded', color='#4299E1', fillcolor='#EBF8FF', 
           penwidth='2.5', fontsize='11', fontcolor='#2C5282', 
           labeljust='c', margin='16')
    
    c.node('Ingest', '📊 Ingestion Agent\nData Cleanse\nType Inference',
           fillcolor='#BEE3F8', color='#3182CE', fontcolor='#1A365D')
    c.node('Fingerprint', '🧬 Data Fingerprint\nSparsity & Noise Metrics\nDimensionality Ratios',
           fillcolor='#90CDF4', color='#2B6CB0', fontcolor='#1A365D', shape='note')
    c.node('DAG', '⚙️ DAG Orchestrator\nDynamic Route Planning\nTask Allocation',
           fillcolor='#63B3ED', color='#2C5282', fontcolor='#1A365D')

# Headquarters (Backend) - The Agentic Execution
with dot.subgraph(name='cluster_backend') as c:
    c.attr(label='🏢 Multi-Agent Execution (DeepSeek-V3.2)', 
           style='filled,rounded', color='#9F7AEA', fillcolor='#FAF5FF',
           penwidth='3', fontsize='12', fontcolor='#553C9A', 
           labeljust='c', margin='20')
    
    c.node('EDA', '📈 EDA Processor\nDirect Pandas Compute\nCorrelation Clustering',
           fillcolor='#E9D8FD', color='#805AD5', fontcolor='#44337A')
    
    c.node('Modeler', '🧠 Modeling Agent\nRL Iteration Management\nFeature Engineering',
           fillcolor='#D6BCFA', color='#6B46C1', fontcolor='#322659',
           penwidth='3', width='3.2', height='1.3')
    
    c.node('Omega', '🚀 DataCortex-Omega\nMeta-Stack Super-Ensemble\n(CatBoost + LGBM + XGB + RF)',
           fillcolor='#ED64A6', color='#D53F8C', fontcolor='#521B41',
           penwidth='3', width='3.5', height='1.3')

# Safety Governance
with dot.subgraph(name='cluster_safety') as c:
    c.attr(label='🛡️ Multi-Judge Isolation Architecture', 
           style='filled,rounded', color='#ED8936', fillcolor='#FFFAF0',
           penwidth='2.5', fontsize='12', fontcolor='#9C4221',
           labeljust='c', margin='16')
    
    c.node('Guardian', '👁️ Guardian Agent\n(Llama-Guard 4)\nCross-Family Validation\nSecurity Governance',
           fillcolor='#FEF089', color='#F6E05E', fontcolor='#744210',
           shape='hexagon', style='filled,rounded', penwidth='2')
           
    c.node('AST', '🔍 AST Parser\nAbstract Syntax Tree Intercept\nBlock Eval/Network Modules',
           fillcolor='#FBD38D', color='#DD6B20', fontcolor='#7C2D12')
    
# Secure Sandboxing
with dot.subgraph(name='cluster_sandbox') as c:
    c.attr(label='☁️ Cloud Sandbox Execution', 
           style='filled,rounded', color='#48BB78', fillcolor='#F0FFF4',
           penwidth='2.5', fontsize='11', fontcolor='#276749',
           labeljust='c', margin='16')
    
    c.node('Sandbox', '⚙️ Blaxel Docker Engine\nIsolated System Compute\nZero-Trust Environment',
           fillcolor='#C6F6D5', color='#38A169', fontcolor='#22543D',
           width='2.8', height='1.6')
           
    c.node('Optuna', '⚙️ Optuna Server\nBayesian HPO Trials\nHyperparameter Tuning',
           fillcolor='#9AE6B4', color='#68D391', fontcolor='#22543D')

# Main Flow Connections
dot.edge('title', 'User', style='invis')
dot.edge('User', 'Ingest', xlabel='Dataset & Task', color='#667EEA', 
         penwidth='2.5', fontcolor='#5A67D8')

dot.edge('Ingest', 'Fingerprint', color='#4299E1', penwidth='2')
dot.edge('Fingerprint', 'DAG', color='#4299E1', penwidth='2')

# DAG Orchestration
dot.edge('DAG', 'EDA', xlabel='Launch', color='#805AD5', penwidth='2.5', fontcolor='#553C9A')
dot.edge('DAG', 'Modeler', xlabel='Launch', color='#805AD5', penwidth='2.5', fontcolor='#553C9A')

# Modeler escalating to Omega
dot.edge('Modeler', 'Omega', xlabel='RL Feedback Iter 3', color='#D53F8C', penwidth='3', style='dashed')

# Modeler proposing code
dot.edge('Omega', 'Guardian', xlabel='Proposed Code', color='#E53E3E', penwidth='2.5', fontcolor='#9B2C2C')
dot.edge('EDA', 'Guardian', color='#E53E3E', penwidth='2.5', fontcolor='#9B2C2C')

dot.edge('Guardian', 'AST', color='#DD6B20', penwidth='2', dir='both')
dot.edge('Guardian', 'Sandbox', xlabel='Approved Code', color='#38A169', penwidth='3', fontcolor='#276749')
dot.edge('Guardian', 'Modeler', xlabel='Safety Reject', color='#E53E3E', penwidth='2', style='dashed', fontcolor='#9B2C2C')

# Sandbox execution
dot.edge('Sandbox', 'Optuna', color='#38A169', penwidth='2', dir='both')
dot.edge('Sandbox', 'DAG', xlabel='Result JSON', color='#38A169', penwidth='2.5', fontcolor='#276749')


# Feature annotations
dot.node('innovation1', '💡 Patent Claim 1\nFingerprint-Guided Planning without Brute Force',
         shape='note', fillcolor='#FFF5F5', color='#FC8181', 
         fontcolor='#742A2A', fontsize='9', style='filled')
dot.node('innovation2', '💡 Patent Claim 4\nMeta-Stack Agentic RL scaling to XGBoost/CatBoost',
         shape='note', fillcolor='#FFFFF0', color='#F6E05E',
         fontcolor='#744210', fontsize='9', style='filled')

dot.edge('DAG', 'innovation1', style='invis')
dot.edge('Omega', 'innovation2', style='invis')


# Render outputs
img_path = os.path.join(OUTPUT_DIR, '05_advanced_architecture_diagram')
dot.render(img_path, format='png', cleanup=True)
dot.render(img_path, format='svg', cleanup=False)

print("✅ DataCortex-AI Advanced Architecture Generated!")
print(f"   📊 {img_path}.png")
print(f"   📊 {img_path}.svg")
print("\n" + "="*70)
