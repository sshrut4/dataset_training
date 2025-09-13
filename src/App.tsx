import React, { useState } from 'react';
import { Download, MessageSquare, Phone, Video, Database, Shield, AlertTriangle, BookOpen } from 'lucide-react';
import TrainingGuide from './components/TrainingGuide';

function App() {
  const [generatedData, setGeneratedData] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('guide');

  const generateTextDataset = () => {
    const scamMessages = [
      {
        channel: "sms",
        sender: "+1-555-0123",
        content: "URGENT: Digital arrest warrant issued against you. Police will arrest you in 2 hours unless you call 1-800-SCAM-123 immediately. Share OTP to verify identity.",
        label: "scam"
      },
      {
        channel: "email",
        sender: "cybercrime.dept@fake-gov.com",
        content: "Your bank account has been linked to money laundering. Immediate arrest warrant issued. Transfer ₹50,000 to account 1234567890 within 1 hour to avoid jail.",
        label: "scam"
      },
      {
        channel: "sms",
        sender: "+91-9999888777",
        content: "POLICE NOTICE: Your Aadhaar linked to illegal activities. Report to station OR transfer security deposit ₹25,000 to avoid arrest. Call 9876543210",
        label: "scam"
      },
      {
        channel: "email",
        sender: "enforcement@cybercrime-delhi.org",
        content: "Legal Notice: Your phone number used for terrorist activities. Digital arrest in progress. Pay fine ₹75,000 to nodalofficer@payment.com immediately.",
        label: "scam"
      },
      {
        channel: "sms",
        sender: "+1-800-FAKE-COP",
        content: "ARREST WARRANT: Your SSN involved in drug trafficking. Share your bank OTP NOW or face immediate imprisonment. Time remaining: 45 minutes.",
        label: "scam"
      },
      {
        channel: "email",
        sender: "digital.arrest@cbi-india.gov.fake",
        content: "CBI URGENT: Your account frozen due to suspicious transactions. Unfreeze by paying verification fee ₹1,00,000 within 2 hours. Reply with card details.",
        label: "scam"
      },
      {
        channel: "sms",
        sender: "+44-7700-900123",
        content: "UK POLICE: Identity theft reported using your details. Avoid prosecution by paying compensation £5,000. WhatsApp: +44-7700-900124",
        label: "scam"
      },
      {
        channel: "email",
        sender: "commissioner@mumbai-police.fake.com",
        content: "Your PAN card misused for hawala transactions. Immediate digital arrest unless you cooperate. Transfer ₹2,00,000 to investigation account within 3 hours.",
        label: "scam"
      },
      {
        channel: "sms",
        sender: "+1-555-ARREST",
        content: "FBI WARNING: Your name in terrorist database. Surrender at nearest station OR pay penalty $10,000 to avoid arrest. Call Agent Johnson: 555-0199",
        label: "scam"
      },
      {
        channel: "email",
        sender: "cybercell@delhi-police.scam.org",
        content: "FINAL NOTICE: Digital arrest warrant active. Your location tracked. Pay ₹3,50,000 settlement amount to avoid police raid at your address.",
        label: "scam"
      },
      {
        channel: "sms",
        sender: "+91-8888777666",
        content: "RBI ALERT: Your account used for money laundering. Deposit ₹5,00,000 in penalty account or face 10-year imprisonment. Act within 1 hour.",
        label: "scam"
      },
      {
        channel: "email",
        sender: "investigation.officer@ncb.fake.in",
        content: "NCB NOTICE: Narcotics purchased using your card. Avoid arrest by paying fine ₹15,00,000. Contact investigating officer at +91-9876543211 immediately.",
        label: "scam"
      },
      {
        channel: "sms",
        sender: "+1-202-FAKE-GOV",
        content: "IRS FINAL NOTICE: Tax evasion case filed. Pay $25,000 penalty OR face federal arrest within 4 hours. Share SSN for verification: 1-800-555-0144",
        label: "scam"
      },
      {
        channel: "email",
        sender: "digitalarrest@income-tax.fake.gov.in",
        content: "Income Tax Dept: Your ITR shows ₹50 lakhs black money. Digital arrest initiated. Deposit penalty ₹10,00,000 to avoid prosecution.",
        label: "scam"
      },
      {
        channel: "sms",
        sender: "+65-8888-9999",
        content: "Singapore Police: Your passport flagged for illegal entry. Pay S$20,000 clearance fee within 2 hours or face international arrest warrant.",
        label: "scam"
      },
      {
        channel: "email",
        sender: "courtorder@supreme-court.fake.in",
        content: "Supreme Court Notice: Your property to be seized for court case. Deposit ₹25,00,000 interim bond to prevent attachment of assets.",
        label: "scam"
      },
      {
        channel: "sms",
        sender: "+91-7777666555",
        content: "ED ENFORCEMENT: Your bank account under investigation for foreign exchange violations. Transfer ₹8,00,000 fine to avoid arrest.",
        label: "scam"
      },
      {
        channel: "email",
        sender: "cybersecurity@reserve-bank.fake.in",
        content: "RBI URGENT: Your account hacked by international fraudsters. Secure account by transferring balance to safety account: 9876543210987654.",
        label: "scam"
      },
      {
        channel: "sms",
        sender: "+1-415-SCAM-CALL",
        content: "US CUSTOMS: Package containing drugs seized with your address. Pay $50,000 penalty within 3 hours or face federal charges. Call: 1-415-555-0188",
        label: "scam"
      },
      {
        channel: "email",
        sender: "investigation@central-bureau.fake.gov.in",
        content: "CBI FINAL WARNING: Your Aadhaar linked to terror funding. Surrender ₹20,00,000 security deposit within 6 hours to avoid life imprisonment.",
        label: "scam"
      }
    ];

    const legitMessages = [
      {
        channel: "sms",
        sender: "HDFC-BANK",
        content: "Your HDFC Bank account ending 1234 has been credited with ₹15,000 on 15-Jan-2025. Available balance: ₹45,000. For help call 18002586161",
        label: "legit"
      },
      {
        channel: "email",
        sender: "noreply@sbi.co.in",
        content: "Dear Customer, Your SBI account statement for December 2024 is ready. Download from SBI Online or visit nearest branch. Customer care: 1800112211",
        label: "legit"
      },
      {
        channel: "sms",
        sender: "UIDAI-GOV",
        content: "Your Aadhaar update request has been processed successfully. Download updated Aadhaar from uidai.gov.in within 90 days. No fees required.",
        label: "legit"
      },
      {
        channel: "email",
        sender: "incometax@gov.in",
        content: "ITR filing reminder: Last date to file Income Tax Return for AY 2024-25 is 31st July 2024. File online at incometaxindia.gov.in to avoid penalty.",
        label: "legit"
      },
      {
        channel: "sms",
        sender: "ICICI-BANK",
        content: "ICICI Bank: Your fixed deposit of ₹1,00,000 matures on 25-Jan-2025. Visit branch or use iMobile app to renew. Customer care: 18001080",
        label: "legit"
      },
      {
        channel: "email",
        sender: "passport@gov.in",
        content: "Passport Application Update: Your passport application (File: AB1234567) is under review. Expected processing time: 30 days. Track status online.",
        label: "legit"
      },
      {
        channel: "sms",
        sender: "AXIS-BANK",
        content: "Axis Bank: Your credit card bill of ₹8,450 is due on 20-Jan-2025. Pay online via Axis Mobile app or net banking. Minimum due: ₹845",
        label: "legit"
      },
      {
        channel: "email",
        sender: "epfo@gov.in",
        content: "EPFO Update: Your PF balance as of Dec 2024 is ₹2,45,000. Annual statement available on epfindia.gov.in. Use UAN for login.",
        label: "legit"
      },
      {
        channel: "sms",
        sender: "KOTAK-BANK",
        content: "Kotak Bank: Thank you for opening savings account. Your account number is 1234567890. Debit card will be dispatched within 7 working days.",
        label: "legit"
      },
      {
        channel: "email",
        sender: "rto@transport.gov.in",
        content: "Driving License Renewal: Your DL expires on 15-Feb-2025. Renew online at parivahan.gov.in or visit RTO office. Required documents listed on website.",
        label: "legit"
      },
      {
        channel: "sms",
        sender: "YES-BANK",
        content: "YES Bank: Your loan EMI of ₹12,500 has been auto-debited from account ending 5678. Next due date: 15-Feb-2025. Outstanding: ₹4,50,000",
        label: "legit"
      },
      {
        channel: "email",
        sender: "voter@eci.gov.in",
        content: "Election Commission: Voter ID card application approved. Download e-EPIC from nvsp.in using reference number VID123456789. Valid across India.",
        label: "legit"
      },
      {
        channel: "sms",
        sender: "CANARA-BANK",
        content: "Canara Bank: Interest of ₹2,100 credited to your savings account for Q3. Current balance: ₹65,400. Thank you for banking with us.",
        label: "legit"
      },
      {
        channel: "email",
        sender: "pgportal@gov.in",
        content: "Grievance Update: Your complaint (ID: PG202412345) regarding pension delay has been forwarded to concerned department. Response expected within 30 days.",
        label: "legit"
      },
      {
        channel: "sms",
        sender: "UNION-BANK",
        content: "Union Bank: Your account has been credited with salary ₹55,000. TDS deducted: ₹8,200. Download salary certificate from UBI mobile app.",
        label: "legit"
      },
      {
        channel: "email",
        sender: "pfrda@gov.in",
        content: "NPS Update: Your National Pension System contribution of ₹5,000 for Jan 2025 has been processed. View statement on npscra.nsdl.co.in",
        label: "legit"
      },
      {
        channel: "sms",
        sender: "BOI-BANK",
        content: "Bank of India: Your term deposit of ₹2,00,000 has matured. Visit branch with passbook to withdraw or renew. Interest earned: ₹14,600",
        label: "legit"
      },
      {
        channel: "email",
        sender: "gst@gov.in",
        content: "GST Return Filing: GSTR-3B for December 2024 is due by 20-Jan-2025. File online at gst.gov.in to avoid late fees. Use GST login credentials.",
        label: "legit"
      },
      {
        channel: "sms",
        sender: "INDIAN-BANK",
        content: "Indian Bank: Your home loan interest certificate for FY 2023-24 is ready. Download from IndOASIS portal for tax filing purposes.",
        label: "legit"
      },
      {
        channel: "email",
        sender: "mygov@gov.in",
        content: "MyGov Alert: New scheme for digital payments launched. Benefits include cashback on UPI transactions. Register at mygov.in for updates and participation.",
        label: "legit"
      }
    ];

    const dataset = [...scamMessages, ...legitMessages];
    setGeneratedData({ type: 'text', data: dataset });
  };

  const generateAudioDataset = () => {
    const scamCalls = [
      {
        caller: "+91-11-26701234 (Delhi Police - FAKE)",
        content: "This is Inspector Sharma from Delhi Police Cyber Crime. Your Aadhaar card has been misused for money laundering. You need to transfer ₹50,000 immediately to investigation account or you will be arrested within 2 hours.",
        label: "scam"
      },
      {
        caller: "+1-202-555-FBI (Federal Bureau - FAKE)",
        content: "FBI Agent Williams speaking. Your SSN is linked to drug trafficking activities. To avoid federal arrest, you must pay a $25,000 clearance fee right now. Do not disconnect this call.",
        label: "scam"
      },
      {
        caller: "+91-22-27431234 (Mumbai Police - FAKE)",
        content: "This is ACP Verma from Mumbai Police. Your bank account shows suspicious transactions worth ₹10 lakhs. Transfer ₹2 lakhs security deposit within one hour or we will freeze all your assets.",
        label: "scam"
      },
      {
        caller: "+44-20-7946-FAKE (Scotland Yard)",
        content: "Metropolitan Police calling. Your passport has been used for illegal entry into UK. Pay £15,000 penalty fee immediately through Western Union or face international arrest warrant.",
        label: "scam"
      },
      {
        caller: "+91-80-22942234 (CBI Bangalore - FAKE)",
        content: "CBI Officer Reddy here. Your PAN card is linked to terrorist financing. You have 30 minutes to deposit ₹5 lakhs in investigation account. Failure to comply will result in immediate arrest.",
        label: "scam"
      },
      {
        caller: "+1-415-555-IRS (Internal Revenue - FAKE)",
        content: "IRS final notice. You owe $50,000 in back taxes and penalties. Pay immediately via iTunes gift cards or face asset seizure and arrest within 24 hours. This is your last warning.",
        label: "scam"
      },
      {
        caller: "+91-33-22834567 (Kolkata Police - FAKE)",
        content: "Senior Inspector Das from Kolkata Police Economic Offences. Your credit card used for hawala transactions. Deposit ₹3 lakhs penalty amount right now or face 7-year imprisonment under PMLA Act.",
        label: "scam"
      },
      {
        caller: "+65-6225-FAKE (Singapore Police)",
        content: "Singapore Police Force. Your work permit application shows false information. Pay S$30,000 fine through bank transfer within 2 hours or face deportation and 5-year entry ban.",
        label: "scam"
      },
      {
        caller: "+91-44-28447890 (Chennai Cyber - FAKE)",
        content: "Cyber Crime Wing Chennai. Your phone number used to send threatening messages to government officials. Transfer ₹8 lakhs to avoid sedition charges. This is non-negotiable.",
        label: "scam"
      },
      {
        caller: "+1-310-555-LAPD (Los Angeles - FAKE)",
        content: "LAPD Detective Johnson. Your identity stolen for drug purchase. To clear your record, send $40,000 via cryptocurrency within 3 hours. Do not involve anyone else in this matter.",
        label: "scam"
      }
    ];

    const legitCalls = [
      {
        caller: "HDFC Bank Customer Care",
        content: "Good morning, this is Priya from HDFC Bank. Your credit card application has been approved. We need to schedule a visit for document verification and card delivery. What time would be convenient for you?",
        label: "legit"
      },
      {
        caller: "Income Tax Department",
        content: "Hello, this is Mr. Kumar from Income Tax Processing Center, Bangalore. Your ITR has been selected for scrutiny assessment. Please visit our office on 25th January with required documents for verification.",
        label: "legit"
      },
      {
        caller: "SBI Home Loans",
        content: "Good afternoon, I'm Ravi from SBI Home Loans. Your loan application is being processed. We need salary certificates for the last 6 months and property documents. Can you visit our branch this week?",
        label: "legit"
      },
      {
        caller: "UIDAI Aadhaar Center",
        content: "This is Mrs. Sharma from Aadhaar Enrollment Center. Your biometric update appointment is scheduled for tomorrow 2 PM. Please bring original PAN card and address proof documents.",
        label: "legit"
      },
      {
        caller: "ICICI Bank Relationship Manager",
        content: "Hi, this is Amit, your relationship manager from ICICI Bank. Your fixed deposit is maturing next week. Would you like to renew it or transfer the amount to your savings account?",
        label: "legit"
      },
      {
        caller: "Passport Seva Kendra",
        content: "Good morning, calling from Regional Passport Office. Your passport application requires police verification. The officer will visit your address on 20th January between 10 AM to 4 PM. Please be available.",
        label: "legit"
      },
      {
        caller: "LIC Insurance Advisor",
        content: "This is Sunita from LIC Branch Office. Your annual premium of ₹25,000 is due next month. You can pay online, visit our office, or I can arrange for home collection. Which option suits you?",
        label: "legit"
      },
      {
        caller: "EPFO Regional Office",
        content: "Hello, this is Mr. Gupta from EPFO. Your PF withdrawal request is being processed. We need your cancelled cheque and Form 15G for tax exemption. Please submit within 15 days.",
        label: "legit"
      },
      {
        caller: "Axis Bank Credit Card",
        content: "This is Neha from Axis Bank Credit Cards. Your card limit enhancement request has been approved. Your new limit is ₹5 lakhs effective immediately. Is there anything else I can help you with?",
        label: "legit"
      },
      {
        caller: "RTO Licensing Section",
        content: "Good afternoon, this is the Regional Transport Office. Your driving license renewal is due next month. You can apply online at Parivahan portal or visit our office with required documents and fees.",
        label: "legit"
      }
    ];

    const dataset = [...scamCalls, ...legitCalls];
    setGeneratedData({ type: 'audio', data: dataset });
  };

  const generateVideoDataset = () => {
    const scamVideos = [
      {
        scene: "Person in police uniform sitting at desk with fake ID badge, Indian flag in background, official-looking room setup",
        dialogue: "I am Joint Commissioner Rajesh Kumar from Delhi Police Cyber Crime. Your Aadhaar card has been compromised and used for illegal activities. You have exactly 2 hours to transfer ₹2 lakh rupees to our investigation account to avoid immediate arrest.",
        label: "scam"
      },
      {
        scene: "Individual in formal attire with fake CBI badge, government office backdrop, official seal visible on desk",
        dialogue: "This is Deputy Director Anil Sharma, Central Bureau of Investigation. Your bank account shows suspicious transactions of ₹15 lakhs. Transfer ₹5 lakhs security deposit immediately or face money laundering charges with 10-year imprisonment.",
        label: "scam"
      },
      {
        scene: "Person in military-style uniform with fake badges, serious expression, official-looking chamber with flags",
        dialogue: "Major General Vikram Singh, Military Intelligence. Your phone number has been linked to anti-national activities. Pay ₹10 lakhs penalty within 4 hours to our secure account or face court martial and life imprisonment.",
        label: "scam"
      },
      {
        scene: "Individual in formal suit with fake Supreme Court backdrop, legal books visible, official atmosphere",
        dialogue: "Justice Praveen Kumar speaking from Supreme Court chambers. Your property documents show fraudulent registration. Deposit ₹25 lakhs interim bond immediately to prevent seizure of all your assets and arrest.",
        label: "scam"
      },
      {
        scene: "Person in banking attire with fake RBI logo, professional office setting, computer screens showing fake data",
        dialogue: "This is Deputy Governor Suresh Patel, Reserve Bank of India. Your account has been flagged for international money laundering. Transfer ₹8 lakhs verification amount within 3 hours or face permanent account freezing and criminal prosecution.",
        label: "scam"
      }
    ];

    const legitVideos = [
      {
        scene: "Bank officer in professional attire sitting at clean desk with HDFC Bank logo, customer service area in background",
        dialogue: "Good morning, I'm Priya Sharma, Branch Manager at HDFC Bank. Your home loan application has been approved for ₹50 lakhs. We need to complete the documentation process. Can you visit our branch this week with the required papers?",
        label: "legit"
      },
      {
        scene: "Government official in formal dress at clean desk with Income Tax Department logo, certificates on wall",
        dialogue: "Hello, this is Mr. Rajesh Kumar from Income Tax Assessment Unit, Mumbai. Your tax return requires some clarifications regarding business income. Please visit our office on 15th February with your books of accounts and supporting documents.",
        label: "legit"
      },
      {
        scene: "Professional in business attire at SBI branch, bank environment with SBI branding, customer service desk",
        dialogue: "I'm Amit Verma, your relationship manager from SBI. Your fixed deposit of ₹5 lakhs is maturing next month. We have several investment options available. Would you like to schedule an appointment to discuss your portfolio?",
        label: "legit"
      },
      {
        scene: "Insurance advisor in professional setting with LIC logo, policy documents visible, corporate office environment",
        dialogue: "This is Mrs. Sunita Reddy from LIC of India. Your annual premium payment is due next month. You can pay online through our portal, visit our office, or I can arrange for premium collection at your residence. Which option would you prefer?",
        label: "legit"
      },
      {
        scene: "Government official at UIDAI office with proper branding, enrollment setup visible, professional government office",
        dialogue: "Good afternoon, I'm calling from UIDAI Aadhaar Center, Sector 15. Your biometric update appointment is confirmed for tomorrow at 11 AM. Please bring original documents for address proof and identity verification as per the list sent via SMS.",
        label: "legit"
      }
    ];

    const dataset = [...scamVideos, ...legitVideos];
    setGeneratedData({ type: 'video', data: dataset });
  };

  const generateUnifiedDataset = () => {
    const mixedDataset = [
      // SMS Scams
      { channel: "sms", sender: "+91-9876543210", content: "URGENT: Your Aadhaar linked to ₹50L fraud. Digital arrest in 2 hrs. Call 1800-123-FAKE immediately.", label: "scam" },
      { channel: "sms", sender: "+1-555-0123", content: "IRS FINAL NOTICE: Pay $25,000 now or face federal arrest. Call 1-800-555-FAKE", label: "scam" },
      
      // Email Scams  
      { channel: "email", sender: "cybercrime@fake-police.com", content: "Police Notice: Your bank account frozen for money laundering. Pay ₹2L fine to unfreeze within 3 hours.", label: "scam" },
      { channel: "email", sender: "cbi.investigation@fake.gov.in", content: "CBI URGENT: Your PAN linked to terror funding. Transfer ₹10L to investigation account immediately.", label: "scam" },
      
      // Call Scams
      { channel: "call", sender: "Delhi Police (FAKE)", content: "Inspector Sharma here. Your account shows illegal transactions. Transfer ₹3L security deposit now or face arrest within 1 hour.", label: "scam" },
      { channel: "call", sender: "FBI Agent (FAKE)", content: "Your SSN compromised for drug trafficking. Pay $50,000 penalty via Bitcoin within 4 hours to avoid federal charges.", label: "scam" },
      
      // Video Scams
      { channel: "video", sender: "CBI Officer (FAKE)", content: "Deputy Director speaking from CBI headquarters. Your documents linked to anti-national activities. Deposit ₹15L penalty immediately.", label: "scam" },
      { channel: "video", sender: "Supreme Court (FAKE)", content: "Justice Kumar from Supreme Court. Your property under seizure order. Pay ₹20L bond within 2 hours to prevent attachment.", label: "scam" },
      
      // More SMS Scams
      { channel: "sms", sender: "+44-7700-900123", content: "UK Border Agency: Your visa application shows fraud. Pay £10,000 penalty within 24 hours or face deportation.", label: "scam" },
      { channel: "sms", sender: "+91-8888777666", content: "RBI Alert: Account hacked by international fraudsters. Transfer funds to safety account 9876543210 immediately.", label: "scam" },
      { channel: "sms", sender: "+1-202-555-0199", content: "US Treasury: Tax evasion detected. Wire $75,000 to IRS account within 6 hours to avoid arrest warrant.", label: "scam" },
      { channel: "sms", sender: "+91-7777666555", content: "ED Notice: Foreign exchange violation. Pay ₹12L penalty via RTGS to avoid 5-year imprisonment under FEMA.", label: "scam" },
      
      // Email Scams continued
      { channel: "email", sender: "mumbai.police@fake.co.in", content: "Commissioner Office: Your credit card used for hawala transactions. Deposit ₹8L investigation fee within 4 hours.", label: "scam" },
      { channel: "email", sender: "income.tax@fake.gov.in", content: "IT Department: Black money found in your returns. Pay ₹25L penalty immediately to avoid prosecution under Income Tax Act.", label: "scam" },
      { channel: "email", sender: "ncb.officer@fake-narcotics.in", content: "NCB URGENT: Drugs purchased using your card details. Transfer ₹18L fine to avoid arrest under NDPS Act.", label: "scam" },
      
      // Legitimate SMS
      { channel: "sms", sender: "HDFC-BANK", content: "Your HDFC account credited with ₹45,000 salary. Available balance: ₹67,500. For support call 18002586161", label: "legit" },
      { channel: "sms", sender: "UIDAI-GOV", content: "Aadhaar update completed successfully. Download from uidai.gov.in using enrollment number 1234/56789/12345", label: "legit" },
      { channel: "sms", sender: "SBI-BANK", content: "SBI Credit card bill ₹15,680 due on 25-Jan-2025. Pay via YONO app or netbanking. Minimum due: ₹1,568", label: "legit" },
      { channel: "sms", sender: "ICICI-BANK", content: "Fixed deposit ₹2,00,000 matures on 30-Jan-2025. Auto-renew enabled. To modify, visit branch or use iMobile app", label: "legit" },
      
      // Legitimate Emails
      { channel: "email", sender: "noreply@incometax.gov.in", content: "ITR Filing Reminder: Last date for AY 2024-25 is 31st July 2024. File online at incometaxindia.gov.in", label: "legit" },
      { channel: "email", sender: "epfo@gov.in", content: "EPFO Statement: Your PF balance as of Dec 2024 is ₹3,45,000. Download from epfindia.gov.in using UAN", label: "legit" },
      { channel: "email", sender: "passport@mea.gov.in", content: "Passport Application Status: Your application AB1234567 is under police verification. Expected completion: 15 days", label: "legit" },
      
      // Legitimate Calls
      { channel: "call", sender: "HDFC Customer Care", content: "This is Priya from HDFC Bank. Your loan application approved. Please visit branch with salary slips and ID proof for documentation.", label: "legit" },
      { channel: "call", sender: "LIC Branch Office", content: "Sunita from LIC calling. Your premium ₹25,000 due next month. You can pay online, visit office, or arrange home collection.", label: "legit" },
      { channel: "call", sender: "SBI Home Loans", content: "Rajesh from SBI Home Loans. Your application is being processed. We need 6 months salary certificates and property documents.", label: "legit" },
      
      // Legitimate Videos  
      { channel: "video", sender: "HDFC Branch Manager", content: "Good morning, I'm the Branch Manager. Your home loan of ₹40L has been sanctioned. Please complete documentation within 15 days.", label: "legit" },
      { channel: "video", sender: "Income Tax Officer", content: "This is Assessment Officer from IT Department. Your return requires clarification on capital gains. Please visit with supporting documents.", label: "legit" },
      { channel: "video", sender: "UIDAI Official", content: "Calling from Aadhaar Center. Your biometric update appointment confirmed for tomorrow 2 PM. Bring original ID and address proof.", label: "legit" },
      
      // Additional Scams
      { channel: "call", sender: "Singapore Police (FAKE)", content: "Your work permit shows false information. Pay S$25,000 fine via bank transfer within 3 hours or face deportation.", label: "scam" },
      { channel: "video", sender: "RBI Governor (FAKE)", content: "This is RBI Governor. Your account flagged for international money laundering. Deposit ₹30L verification fee immediately.", label: "scam" },
      { channel: "email", sender: "supreme.court@fake-justice.in", content: "Court Order: Your assets under attachment. Pay ₹50L court fee within 24 hours to prevent seizure of property.", label: "scam" }
    ];

    setGeneratedData({ type: 'unified', data: mixedDataset });
  };

  const downloadDataset = () => {
    if (!generatedData) return;
    
    const dataStr = JSON.stringify(generatedData.data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${generatedData.type}-dataset.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const tabs = [
    { id: 'guide', label: 'Training Guide', icon: BookOpen },
    { id: 'text', label: 'Text (SMS/Email)', icon: MessageSquare },
    { id: 'audio', label: 'Audio (Calls)', icon: Phone },
    { id: 'video', label: 'Video Calls', icon: Video },
    { id: 'unified', label: 'Mixed Dataset', icon: Database }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-red-500 p-2 rounded-lg">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Digital Arrest Scam Dataset Generator</h1>
                <p className="text-gray-600">Generate realistic datasets for training ML models to detect digital arrest scams</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-500">
              <AlertTriangle className="w-4 h-4 text-amber-500" />
              <span>For Research & Training Only</span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tab Navigation */}
        <div className="mb-8">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                      activeTab === tab.id
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>
        </div>

        {/* Training Guide */}
        {activeTab === 'guide' && <TrainingGuide />}

        {/* Dataset Generation Content */}
        {activeTab !== 'guide' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Panel - Generator */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-sm border p-6 sticky top-6">
              {activeTab === 'text' && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-gray-900">Text Dataset Generator</h3>
                  <p className="text-sm text-gray-600">
                    Generate 40 examples (20 scam, 20 legitimate) of SMS and email messages for training text classification models.
                  </p>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-2">Includes:</h4>
                    <ul className="text-sm text-gray-600 space-y-1">
                      <li>• Scam messages with threats and urgency</li>
                      <li>• Legitimate bank and government communications</li>
                      <li>• Channel, sender, content, and label fields</li>
                      <li>• Ready for JSON processing</li>
                    </ul>
                  </div>
                  <button
                    onClick={generateTextDataset}
                    className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 transition-colors"
                  >
                    Generate Text Dataset
                  </button>
                </div>
              )}

              {activeTab === 'audio' && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-gray-900">Audio Dataset Generator</h3>
                  <p className="text-sm text-gray-600">
                    Generate 20 phone call transcripts (10 scam, 10 legitimate) ready for text-to-speech conversion.
                  </p>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-2">Includes:</h4>
                    <ul className="text-sm text-gray-600 space-y-1">
                      <li>• Scam calls impersonating authorities</li>
                      <li>• Legitimate bank and government calls</li>
                      <li>• Short dialogues (2-3 sentences)</li>
                      <li>• TTS-ready format with caller info</li>
                    </ul>
                  </div>
                  <button
                    onClick={generateAudioDataset}
                    className="w-full bg-green-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-green-700 transition-colors"
                  >
                    Generate Audio Dataset
                  </button>
                </div>
              )}

              {activeTab === 'video' && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-gray-900">Video Dataset Generator</h3>
                  <p className="text-sm text-gray-600">
                    Generate 10 video call scripts (5 scam, 5 legitimate) for deepfake detection training.
                  </p>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-2">Includes:</h4>
                    <ul className="text-sm text-gray-600 space-y-1">
                      <li>• Scam videos with fake uniforms/IDs</li>
                      <li>• Legitimate office video calls</li>
                      <li>• Scene descriptions and dialogues</li>
                      <li>• Visual context for deepfake detection</li>
                    </ul>
                  </div>
                  <button
                    onClick={generateVideoDataset}
                    className="w-full bg-purple-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-purple-700 transition-colors"
                  >
                    Generate Video Dataset
                  </button>
                </div>
              )}

              {activeTab === 'unified' && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-gray-900">Unified Dataset Generator</h3>
                  <p className="text-sm text-gray-600">
                    Generate 30 mixed communication samples across all channels for comprehensive model training.
                  </p>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-2">Includes:</h4>
                    <ul className="text-sm text-gray-600 space-y-1">
                      <li>• 50% scam, 50% legitimate samples</li>
                      <li>• SMS, email, call, and video formats</li>
                      <li>• Cross-channel pattern recognition</li>
                      <li>• Unified JSON structure</li>
                    </ul>
                  </div>
                  <button
                    onClick={generateUnifiedDataset}
                    className="w-full bg-orange-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-orange-700 transition-colors"
                  >
                    Generate Unified Dataset
                  </button>
                </div>
              )}

              {generatedData && (
                <div className="mt-6 pt-6 border-t">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-medium text-gray-700">
                      Generated: {generatedData.data.length} samples
                    </span>
                    <span className="text-xs text-gray-500">
                      Ready for download
                    </span>
                  </div>
                  <button
                    onClick={downloadDataset}
                    className="w-full bg-gray-800 text-white py-2 px-4 rounded-lg font-medium hover:bg-gray-900 transition-colors flex items-center justify-center space-x-2"
                  >
                    <Download className="w-4 h-4" />
                    <span>Download JSON</span>
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Preview */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-sm border">
              <div className="p-6 border-b">
                <h3 className="text-lg font-semibold text-gray-900">Dataset Preview</h3>
                <p className="text-sm text-gray-600 mt-1">
                  Generated samples will appear here. Click generate to see the data structure.
                </p>
              </div>
              
              <div className="p-6">
                {generatedData ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="bg-green-100 text-green-800 text-xs font-medium px-2.5 py-0.5 rounded">
                          {generatedData.data.filter((item: any) => item.label === 'legit').length} Legitimate
                        </span>
                        <span className="bg-red-100 text-red-800 text-xs font-medium px-2.5 py-0.5 rounded">
                          {generatedData.data.filter((item: any) => item.label === 'scam').length} Scam
                        </span>
                      </div>
                      <span className="text-sm text-gray-500">
                        Total: {generatedData.data.length} samples
                      </span>
                    </div>
                    
                    <div className="bg-gray-900 rounded-lg p-4 overflow-auto max-h-96">
                      <pre className="text-green-400 text-xs font-mono whitespace-pre-wrap">
                        {JSON.stringify(generatedData.data.slice(0, 3), null, 2)}
                        {generatedData.data.length > 3 && (
                          <span className="text-gray-400">
                            {'\n... and '}{generatedData.data.length - 3}{' more samples'}
                          </span>
                        )}
                      </pre>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Database className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                    <p className="text-gray-500">No data generated yet</p>
                    <p className="text-sm text-gray-400 mt-1">
                      Select a dataset type and click generate to see samples
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
        )}

        {/* Footer Info */}
        {activeTab !== 'guide' && (
          <div className="mt-12 bg-amber-50 border border-amber-200 rounded-lg p-6">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5" />
            <div>
              <h4 className="font-semibold text-amber-800 mb-2">Important Usage Guidelines</h4>
              <div className="text-sm text-amber-700 space-y-1">
                <p>• This dataset is generated for research and educational purposes only</p>
                <p>• Do not use these examples to create actual scam communications</p>
                <p>• Ensure proper data handling and privacy measures when training models</p>
                <p>• Consider augmenting with real-world data for production deployment</p>
                <p>• Test models thoroughly before deployment to avoid false positives</p>
              </div>
            </div>
          </div>
        </div>
        )}
      </div>
    </div>
  );
}

export default App;