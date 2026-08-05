"""Document loading, chunking, and synthetic knowledge base generation."""

import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.llm.config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

# =====================================================================
# SYNTHETIC POLICY TEXTS (detailed content for realistic RAG queries)
# =====================================================================

RBI_FRAUD_GUIDELINES_TEXT = """
RBI MASTER DIRECTION ON FRAUD IN BANKS - CLASSIFICATION AND REPORTING GUIDELINES
Section 1: Classification of Fraud
Fraudulent activities in banking institutions are classified under specific categories to streamline tracking, reporting, and resolution. These categories include:
1. Misappropriation and criminal breach of trust: Instances where bank employees, agents, or contractors divert customer or bank funds for unauthorized personal or business use.
2. Fraudulent encashment through forged instruments: Using forged cheques, demand drafts, or pay orders to illegally withdraw funds from accounts.
3. Manipulation of books of account or through fictitious accounts: Creating dummy accounts or falsifying balances to conceal shortfalls, thefts, or unauthorized financial transactions.
4. Negligence and cash shortages: Major cash deficits due to negligence or deliberate fraud.
5. Cheating and forgery: Using fake KYC documentation, false collateral representations, or forged signatures to acquire credit facilities.
6. Unauthorised credit facilities extended for reward: Extending loan or credit limits to unapproved borrowers in exchange for personal financial kickbacks.
7. Unauthorized electronic transactions: Fraudulent card-not-present (CNP) transactions, phishing attacks, SIM swapping, and UPI spoofing.

Section 2: Reporting Timelines and Regulatory Escalation
Under RBI guidelines, prompt escalation of detected fraud is mandatory to prevent systemic risk:
- Report to RBI: Banks must report all fraud cases involving Rs 1 lakh (100,000 INR) and above to the RBI's Central Fraud Registry (CFR) within 3 weeks (21 days) of detection.
- Fraud Monitoring Returns (FMR): Submission of FMR must be completed electronically via the RBI's designated reporting portal.
- Police / CBI Reporting: For public sector banks, cases involving fraud above Rs 3 crore must be reported to the CBI. Smaller cases are reported to the local police or State Cyber Crime cells.
- Internal Escalation: Any suspicious activity exceeding Rs 50,000 must be escalated to the bank's internal Fraud Risk Management Committee (FRMC) within 24 hours.

Section 3: Digital Fraud and Card-Not-Present (CNP) Transactions
RBI directives emphasize the protection of retail electronic payment channels:
- Multi-Factor Authentication (MFA): All CNP transactions (such as e-commerce card payments) must be secured by an additional factor of authentication (AFA), typically a dynamic One-Time Password (OTP) sent to the registered mobile number or biometric validation.
- UPI Security Limits: First-time UPI registrations must restrict transaction limits to Rs 5,000 for the first 24 hours to prevent immediate cash-out from compromised credentials.
- Suspend Thresholds: Online accounts must be automatically suspended if velocity-based alerts are triggered. Velocity anomalies include:
  1. Transaction amount exceeds 10 times the average transaction value of the user's historical 3-month profile.
  2. More than 5 distinct transaction attempts to different UPI handles within a 1-hour window.
  3. Immediate transfer of 95% of account balance within 15 minutes of updating security credentials.

Section 4: Staff Accountability and Forensic Review
Upon classification of a transaction as fraud, the bank must:
- Formulate a Staff Accountability Committee to review the involvement of internal operators. The review must be finalized within 6 months.
- Implement a forensic audit for accounts exceeding Rs 50 crore.
- Establish regular cybersecurity reviews and audit system access logs.
"""

PCI_DSS_REQUIREMENTS_TEXT = """
PCI DSS v4.0 SUMMARY - CARDHOLDER DATA ENVIRONMENT SECURITY STANDARDS
Section 1: Authentication and Access Controls (Requirement 8)
Requirement 8 focuses on identifying users and authenticating access to system components:
- Unique Identification: Assign a unique ID to all individuals with access to the Cardholder Data Environment (CDE). Under no circumstances are shared or group accounts allowed.
- Multi-Factor Authentication (MFA): Implement MFA for all administrative access and remote access to the CDE. This includes access from outside the bank's network as well as internal network segments.
- Password Complexity: Passwords must contain at least 12 characters, including numbers, upper/lower case letters, and special characters. Passwords must be changed every 90 days.
- Lockout Policy: Lock accounts after a maximum of 6 failed login attempts. The lockout duration must be set to at least 30 minutes or require administrator unlock.

Section 2: Logging and Monitoring (Requirement 10)
Requirement 10 details audit trail tracking requirements:
- Log everything: Track and monitor all access to system components, especially cardholder data. Log entries must capture:
  1. User identification (who did it).
  2. Type of event (login, file read, write, execution).
  3. Date and time of transaction.
  4. Success or failure status.
  5. Origin of the event (IP address, system component ID).
- Daily Log Review: Review logs for CDE components daily to detect suspicious behaviors. Automated analysis tools must raise immediate alerts on log modifications or clearance events.
- Audit Trail Retention: Retain CDE audit logs for at least one year, with a minimum of three months' logs immediately available for online analysis.

Section 3: Card-Testing Attacks and Velocity Monitoring
Card-testing is a method where fraudsters use automated scripts to test lists of stolen credit card numbers against merchant accounts:
- Attack Pattern: Characterized by rapid, sequential e-commerce transactions of very small amounts (often Rs 10 to Rs 100) at unusual hours, testing validity before executing large fraud sweeps.
- Velocity Checks: Systems must trigger automated block actions if:
  1. The same card number is used more than 3 times within 10 minutes (regardless of success/failure).
  2. The same IP address initiates card submissions from more than 2 distinct card numbers in a 1-hour window.
- Merchant Category Code (MCC) Monitoring: Establish specific controls for high-risk MCCs, including digital wallets, gaming sites, telecommunications recharge portals, and gift card sellers.

Section 4: Geographic and Device Anomaly Detection
Compliance mandates identifying anomalous location discrepancies:
- Impossible Travel: Flag transactions occurring at physical merchant locations separated by geographic distances that are physically impossible to travel in the elapsed time.
- Device Fingerprinting: Track and block requests where the device fingerprint changes rapidly or indicates an emulator or proxy network (VPN) masking the true source.
"""

FRAUD_PATTERNS_GUIDE_TEXT = """
FINANCIAL FRAUD PATTERNS AND BEHAVIORAL ANALYTICS GUIDE
Pattern 1: Card-Testing Attacks
Card-testing is the primary reconnaissance pattern used by global card fraud syndicates:
- Characteristics: Fraudsters execute small verification charges (often less than ₹50 or $1) to check if the stolen credentials are valid.
- Indicator: Multiple micro-transactions within a very short duration at online payment gateways. Once a test transaction succeeds, it is immediately followed by a large transaction (usually 10x to 100x the test amount) to maximize cash extraction before the cardholder freezes the card.
- Thresholds: Flag any account that exhibits more than 3 micro-transactions (under ₹100) followed by a high-value transaction (over ₹10,000) within 15 minutes.

Pattern 2: Account Takeover (ATO)
Account takeover represents a sudden deviation from historical customer behavior:
- Characteristics: The account changes hands via phishing, SIM swapping, or malware injection.
- Indicators: A sudden change in device ID, browser type, or IP location, combined with instant high-value fund transfers.
- Geographic Anomaly: A transaction is processed from a location physically distant from the location of the previous transaction. For example, a card transaction at a local supermarket in Mumbai, followed by an online transaction from an IP address in Russia or Nigeria 15 minutes later.
- Merchant Anomaly: Transactions at high-risk merchant categories (UPI recharge, gold sellers, electronic shops) that do not align with the customer's historical category distribution.

Pattern 3: Synthetic Identity Fraud
Synthetic identity fraud involves creating fake bank accounts using a mix of real and fake KYC info:
- Characteristics: Newly registered accounts that exhibit immediate high-velocity transaction flows.
- Indicators: The account has no historical profile. Once activated, it immediately receives large UPI deposits followed by rapid withdrawals via ATMs or cash transfers.
- Behavior: Transaction patterns show round sums (e.g., exactly ₹50,000, ₹100,000) to clear limits quickly.

Pattern 4: Step-Up Velocity
Step-up velocity is a behavioral fraud technique designed to bypass basic static limits:
- Characteristics: The transaction amounts increase progressively in a short window.
- Indicator: Day 1: ₹1,000; Day 2: ₹5,000; Day 3: ₹50,000. This "warms up" the fraud monitoring system before executing the final drain.

Pattern 5: High-Risk Anomalies (Night-time and Round Amounts)
- Night-time Fraud: Statistics show that fraudulent transactions are disproportionately concentrated between midnight (12:00 AM) and 6:00 AM. This is when cardholders are asleep and unlikely to block their cards immediately.
- Round Amount Fraud: Genuine retail transactions usually contain decimal parts or odd numbers (e.g. ₹943.50). Fraudulent transactions often use exact round amounts (e.g., ₹20,000, ₹50,000) to maximize extraction bounds.
- Cross-Border Anomalies: International e-commerce transactions occurring on cards that have never been used outside of India.
"""

FINSHIELD_SYSTEM_GUIDE_TEXT = """
FINSHIELD AI FRAUD DETECTION - SYSTEM OPERATING GUIDE
Section 1: Machine Learning Model Layout
FinShield implements a multi-layered classification system to balance detection latency and accuracy:
1. XGBoost Model: Evaluates static, aggregate transaction features (Amount, z-scores, night-time indicators, V features interactions). Optimized for microsecond latency.
2. LightGBM Model: Used in parallel with XGBoost to provide an ensemble vote, utilizing fast leaf-wise tree growth to detect complex non-linear combinations.
3. Isolation Forest: Unsupervised anomaly detector trained exclusively on legitimate transaction volumes. Flags outliers that deviate from the normal distribution.
4. BiLSTM Model: Processes sequences of the last 10 transaction vectors for a user to identify rolling time-series velocity patterns.

Section 2: Interpreting SHAP and Explaining Flags
SHAP (SHapley Additive exPlanations) values measure the feature contribution to the model's prediction:
- positive SHAP value: The feature pushed the model output towards classifying the transaction as FRAUD.
- negative SHAP value: The feature pushed the classification towards LEGITIMATE.
- Key features like V14 and V4 represent latent PCA projections that carry high correlation with historic fraud alerts.
- Amount_zscore represents how many standard deviations the transaction amount is from the user's mean spending profile.

Section 3: Threshold and Escalation Procedures
FinShield operates on two key alert thresholds:
- Standard Block Threshold (0.5): If the ensemble fraud probability is 0.5 or above, the system automatically blocks the transaction and sends an SMS OTP for step-up verification.
- Review Threshold (0.3): If the score is between 0.3 and 0.5, the transaction is allowed to proceed but is routed to the Manual Review Queue.
- Escalation: Flagged high-risk events (score > 0.8) must be escalated to the Fraud Investigation Team.
- False Positives: If a blocked transaction is confirmed as legitimate by the customer, mark it as resolved and add the device ID to the whitelist database.
"""


def create_synthetic_policy_docs() -> None:
    """Creates the data/policy_docs directory and writes 4 synthetic policy files."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    docs_to_write = {
        "rbi_fraud_guidelines.txt": RBI_FRAUD_GUIDELINES_TEXT,
        "pci_dss_requirements.txt": PCI_DSS_REQUIREMENTS_TEXT,
        "fraud_patterns_guide.txt": FRAUD_PATTERNS_GUIDE_TEXT,
        "finshield_system_guide.txt": FINSHIELD_SYSTEM_GUIDE_TEXT
    }
    
    for filename, text in docs_to_write.items():
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text.strip())
            
    print(f"Created 4 synthetic policy files under {DOCS_DIR}")


def load_documents() -> list:
    """Loads all text documents from the policy docs directory using TextLoader.

    Returns:
        List of LangChain Document objects.
    """
    text_files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    documents = []
    
    for filepath in text_files:
        loader = TextLoader(filepath, encoding="utf-8")
        documents.extend(loader.load())
        
    print(f"Loaded {len(text_files)} documents from {DOCS_DIR}")
    return documents


def chunk_documents(documents: list) -> list:
    """Splits Document objects into smaller text chunks.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks
