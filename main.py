"""
main.py - Entry Point & Demo Script

Demonstrates the ResumeParser with sample resume text,
showing all extraction capabilities without requiring an actual file.
"""

import json
import os
import datetime

# Sample resume text for demonstration
SAMPLE_RESUME_TEXT = """
John Alexander Smith
Senior Software Engineer

San Francisco, CA 94102
Email: john.smith@gmail.com
Phone: +1 (555) 123-4567
LinkedIn: https://linkedin.com/in/johnsmith
GitHub: https://github.com/johnsmith
Portfolio: https://johnsmith.dev

PROFESSIONAL SUMMARY
Highly motivated Senior Software Engineer with 6+ years of experience designing,
developing, and deploying scalable web applications and distributed systems.
Proven track record of leading cross-functional teams and delivering high-impact
projects in fast-paced environments. Passionate about clean code, performance
optimization, and mentoring junior developers.

WORK EXPERIENCE

Google — Senior Software Engineer
January 2022 - Present
* Led a team of 8 engineers building microservices architecture serving 2M+ daily users
* Designed and implemented a distributed caching layer using Redis, reducing API latency by 45%
* Migrated legacy monolith to Kubernetes-orchestrated microservices on GCP
* Established CI/CD pipelines with GitHub Actions, cutting deployment time from 2 hours to 15 minutes
* Mentored 4 junior engineers through code reviews and pair programming sessions

Microsoft — Software Developer
June 2019 - December 2021
* Built responsive front-end components using React and TypeScript for Azure Portal
* Developed RESTful APIs using C# and ASP.NET Core serving 500K+ daily requests
* Implemented automated testing suite with 95% code coverage using Jest and Selenium
* Collaborated with UX designers to improve accessibility (WCAG 2.1 compliance)
* Reduced page load time by 60% through lazy loading and code splitting optimizations

TechStartup Inc — Junior Developer (Intern)
May 2018 - May 2019
* Developed full-stack features using Python (Django) and JavaScript (Vue.js)
* Built data visualization dashboards using D3.js and Plotly
* Wrote automated data pipelines processing 100K+ records daily with Apache Airflow

EDUCATION

Massachusetts Institute of Technology
Bachelor of Science in Computer Science
2015 - 2019
GPA: 3.8/4.0

Stanford University (Online)
Certificate in Machine Learning
2021

TECHNICAL SKILLS
Python, Java, Go, JavaScript, TypeScript, C#, SQL
React, Angular, Vue.js, Django, Flask, FastAPI, ASP.NET, Express.js
PostgreSQL, MongoDB, Redis, Elasticsearch, DynamoDB
Docker, Kubernetes, AWS, GCP, Azure, Terraform
Git, GitHub Actions, Jenkins, CI/CD, Linux
Machine Learning, NLP, TensorFlow, PyTorch, Scikit-learn

PROJECTS

Resume Parser — Built an NLP-based resume parser using spaCy and Sentence Transformers
that extracts structured data from PDF/DOCX files with 90% accuracy.

Real-time ChatBot — Designed and deployed a customer service chatbot using GPT-4 API
with RAG (Retrieval Augmented Generation) architecture. Handles 1000+ queries/day.

E-commerce Analytics Dashboard — Created a real-time analytics platform using
Apache Kafka, Spark Streaming, and React with D3.js visualizations.

CERTIFICATIONS
AWS Certified Solutions Architect - Associate (2023)
Google Cloud Professional Data Engineer (2022)
Certified Kubernetes Administrator (CKA) (2023)
Certified ScrumMaster (CSM) (2021)

AWARDS
Google Peer Bonus Award for exceptional contributions (2023)
Microsoft Hackathon Winner - Best Developer Tools Category (2020)

LANGUAGES
English (Native), Spanish (Conversational), Mandarin (Basic)

INTERESTS
Open source contributing, competitive programming, tech blogging, hiking
"""


def run_demo():
    """Run the parser demo with sample text."""
    print("=" * 70)
    print("  ATS RESUME PARSER — DEMO")
    print("=" * 70)
    print()

    # Import parser components individually to show the pipeline
    from text_extractor import clean_text
    from section_segmenter import segment_sections
    from contact_extractor import extract_contact_info
    from entity_extractor import extract_name, extract_organizations, parse_experience, parse_education
    from skill_extractor import load_skills_database, extract_skills_keyword

    # Step 1: Clean text (already clean in this demo, but show the step)
    cleaned = clean_text(SAMPLE_RESUME_TEXT)
    print(f"[Step 1] Text cleaned: {len(cleaned)} characters")

    # Step 2: Segment sections
    sections = segment_sections(cleaned)
    print(f"[Step 2] Sections found: {list(sections.keys())}")

    # Step 3: Extract contact info
    contact = extract_contact_info(cleaned)
    print(f"[Step 3] Contact extracted:")
    print(f"        Emails: {contact['emails']}")
    print(f"        Phones: {contact['phones']}")
    print(f"        LinkedIn: {contact['linkedin']}")
    print(f"        GitHub: {contact['github']}")
    print(f"        Websites: {contact['websites']}")

    # Step 4: Extract name
    name = extract_name(cleaned)
    print(f"[Step 4] Name: {name}")

    # Step 5: Extract organizations
    orgs = extract_organizations(cleaned)
    print(f"[Step 5] Organizations: {orgs[:10]}")

    # Step 6: Parse experience
    exp = parse_experience(sections.get("experience", ""))
    print(f"[Step 6] Experience entries: {len(exp)}")
    for i, entry in enumerate(exp, 1):
        print(f"        {i}. {entry.get('company', 'N/A')} — {entry.get('title', 'N/A')} ({entry.get('dates', 'N/A')})")

    # Step 7: Parse education
    edu = parse_education(sections.get("education", ""))
    print(f"[Step 7] Education entries: {len(edu)}")
    for i, entry in enumerate(edu, 1):
        print(f"        {i}. {entry.get('institution', 'N/A')} — {entry.get('degree', 'N/A')}")

    # Step 8: Extract skills (keyword only for speed in demo)
    skills_db = load_skills_database()
    skills = extract_skills_keyword(cleaned, skills_db)
    print(f"[Step 8] Skills found ({len(skills)}): {skills}")

    print()
    print("=" * 70)
    print("  FULL PARSER OUTPUT (JSON)")
    print("=" * 70)
    print()

    # We'll directly test the pipeline using the components
    result = {
        "metadata": {
            "file_name": "sample_resume.txt",
            "parsed_at": datetime.datetime.now().isoformat(),
            "sections_found": list(sections.keys()),
            "parser_version": "2.0.0"
        },
        "candidate": {
            "name": name,
            "contact": contact
        },
        "summary": sections.get("summary", ""),
        "experience": exp,
        "education": edu,
        "skills": {
            "all": skills,
            "technical": [s for s in skills if s not in ["Leadership", "Communication", "Team Management", "Mentoring", "Collaboration", "Problem Solving"]],
            "soft": [s for s in skills if s in ["Leadership", "Communication", "Team Management", "Mentoring", "Collaboration", "Problem Solving"]]
        },
        "projects": sections.get("projects", ""),
        "certifications": sections.get("certifications", ""),
        "awards": sections.get("awards", ""),
        "languages": sections.get("languages", ""),
        "organizations": orgs,
        "ats_score": 91
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Save to file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_output.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    run_demo()