#!/usr/bin/env python3
"""
ChatReport Lite - Fast Prototype
A lightweight conversational symptom reporting chatbot for head and neck cancer patients
"""

from datetime import datetime
from typing import Dict, Optional

class ChatReportLite:
    """Lightweight version of ChatReport - fast execution, no delays"""
    
    def __init__(self):
        self.symptoms = {}
        self.patient_name = None
        self.conversation_log = []
        
    def log(self, speaker: str, message: str):
        """Log conversation without displaying"""
        self.conversation_log.append({
            'speaker': speaker,
            'message': message,
            'time': datetime.now().strftime("%H:%M:%S")
        })
    
    def bot_say(self, message: str):
        """Display bot message immediately"""
        print(f"\n🤖 ChatReport: {message}")
        self.log('bot', message)
    
    def get_input(self, prompt: str = "") -> str:
        """Get user input"""
        if prompt:
            print(f"\n{prompt}")
        user_input = input("You: ").strip()
        self.log('user', user_input)
        return user_input
    
    def yes_no_check(self, response: str) -> bool:
        """Quick check if response indicates presence of symptom"""
        response_lower = response.lower()
        no_words = ['no', 'not', 'none', "haven't", "doesn't", 'nope', 'negative']
        return not any(word in response_lower for word in no_words)
    
    def run(self):
        """Main conversation flow"""
        print("\n" + "="*60)
        print("  CHATREPORT - Symptom Reporting System")
        print("  Fox Chase Cancer Center")
        print("="*60)
        
        # Greeting
        self.bot_say("Hello! I'm ChatReport. I'll help you share how you've been feeling before your appointment.")
        self.bot_say("This takes about 10 minutes. What's your first name?")
        
        self.patient_name = self.get_input()
        self.bot_say(f"Thanks, {self.patient_name}! Let's get started.")
        
        # Pain Assessment
        self.assess_pain()
        
        # Mouth Symptoms
        self.assess_mouth()
        
        # Swallowing
        self.assess_swallowing()
        
        # Nutrition
        self.assess_nutrition()
        
        # Mood
        self.assess_mood()
        
        # Closing
        self.bot_say("Is there anything else you'd like your doctor to know?")
        additional = self.get_input()
        if additional and len(additional) > 3:
            self.symptoms['additional_notes'] = additional
            self.bot_say("Got it. I've added that to your report.")
        
        self.bot_say(f"Thank you, {self.patient_name}! Your report is ready for your doctor.")
        
        # Generate and show report
        print("\n" + "="*60)
        print(self.generate_report())
        print("="*60)
    
    def assess_pain(self):
        """Pain assessment"""
        self.bot_say("Have you had any pain since your last appointment?")
        response = self.get_input()
        
        if not self.yes_no_check(response):
            self.symptoms['pain'] = {'present': False}
            self.bot_say("Good to hear.")
            return
        
        # Pain is present
        self.symptoms['pain'] = {'present': True, 'details': {}}
        
        self.bot_say("Where is the pain located?")
        location = self.get_input()
        self.symptoms['pain']['details']['location'] = location
        
        self.bot_say("On a scale of 0-10, how bad is it at its worst?")
        severity = self.get_input()
        self.symptoms['pain']['details']['severity'] = severity
        
        self.bot_say("How often do you have this pain?")
        frequency = self.get_input()
        self.symptoms['pain']['details']['frequency'] = frequency
        
        self.bot_say("What are you doing to manage it?")
        management = self.get_input()
        self.symptoms['pain']['details']['management'] = management
    
    def assess_mouth(self):
        """Mouth symptoms assessment"""
        self.bot_say("Have you noticed any dry mouth, mouth sores, or taste changes?")
        response = self.get_input()
        
        if not self.yes_no_check(response):
            self.symptoms['mouth'] = {'present': False}
            return
        
        self.symptoms['mouth'] = {'present': True, 'details': {}}
        
        self.bot_say("Can you describe what you're experiencing?")
        description = self.get_input()
        self.symptoms['mouth']['details']['description'] = description
        
        self.bot_say("How severe is it? (mild, moderate, or severe)")
        severity = self.get_input()
        self.symptoms['mouth']['details']['severity'] = severity
    
    def assess_swallowing(self):
        """Swallowing assessment"""
        self.bot_say("Any difficulty swallowing?")
        response = self.get_input()
        
        if not self.yes_no_check(response):
            self.symptoms['swallowing'] = {'present': False}
            return
        
        self.symptoms['swallowing'] = {'present': True, 'details': {}}
        
        self.bot_say("Tell me more about the difficulty.")
        description = self.get_input()
        self.symptoms['swallowing']['details']['description'] = description
        
        self.bot_say("What types of foods can you eat? (regular, soft, liquids, etc.)")
        food_types = self.get_input()
        self.symptoms['swallowing']['details']['food_types'] = food_types
    
    def assess_nutrition(self):
        """Nutrition assessment"""
        self.bot_say("How has your appetite been?")
        appetite = self.get_input()
        self.symptoms['nutrition'] = {'details': {'appetite': appetite}}
        
        self.bot_say("Have you noticed any weight changes?")
        weight = self.get_input()
        self.symptoms['nutrition']['details']['weight'] = weight
    
    def assess_mood(self):
        """Mood assessment"""
        self.bot_say("How would you describe your mood lately?")
        mood = self.get_input()
        self.symptoms['mood'] = {'details': {'mood': mood}}
        
        # Check for concerning words
        concern_words = ['worried', 'anxious', 'depressed', 'sad', 'down', 'hopeless']
        if any(word in mood.lower() for word in concern_words):
            self.bot_say("Are you having trouble sleeping?")
            sleep = self.get_input()
            self.symptoms['mood']['details']['sleep'] = sleep
    
    def generate_report(self) -> str:
        """Generate clinical report"""
        report = []
        report.append("\nCHATREPORT SYMPTOM SUMMARY")
        report.append("="*60)
        report.append(f"Patient: {self.patient_name}")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("="*60)
        report.append("")
        
        # Pain
        if self.symptoms.get('pain', {}).get('present'):
            report.append("🔴 PAIN - PRESENT")
            details = self.symptoms['pain']['details']
            report.append(f"  Location: {details.get('location', 'N/A')}")
            report.append(f"  Severity: {details.get('severity', 'N/A')}/10")
            report.append(f"  Frequency: {details.get('frequency', 'N/A')}")
            report.append(f"  Management: {details.get('management', 'N/A')}")
        else:
            report.append("✅ Pain - Not reported")
        report.append("")
        
        # Mouth
        if self.symptoms.get('mouth', {}).get('present'):
            report.append("🔴 MOUTH SYMPTOMS - PRESENT")
            details = self.symptoms['mouth']['details']
            report.append(f"  Description: {details.get('description', 'N/A')}")
            report.append(f"  Severity: {details.get('severity', 'N/A')}")
        else:
            report.append("✅ Mouth - Not reported")
        report.append("")
        
        # Swallowing
        if self.symptoms.get('swallowing', {}).get('present'):
            report.append("🔴 SWALLOWING DIFFICULTY - PRESENT")
            details = self.symptoms['swallowing']['details']
            report.append(f"  Description: {details.get('description', 'N/A')}")
            report.append(f"  Food types: {details.get('food_types', 'N/A')}")
        else:
            report.append("✅ Swallowing - No difficulty")
        report.append("")
        
        # Nutrition
        if 'nutrition' in self.symptoms:
            report.append("📊 NUTRITION")
            details = self.symptoms['nutrition']['details']
            report.append(f"  Appetite: {details.get('appetite', 'N/A')}")
            report.append(f"  Weight: {details.get('weight', 'N/A')}")
        report.append("")
        
        # Mood
        if 'mood' in self.symptoms:
            report.append("💭 MOOD")
            details = self.symptoms['mood']['details']
            report.append(f"  Mood: {details.get('mood', 'N/A')}")
            if 'sleep' in details:
                report.append(f"  Sleep: {details['sleep']}")
        report.append("")
        
        # Additional notes
        if 'additional_notes' in self.symptoms:
            report.append("📝 ADDITIONAL NOTES")
            report.append(f"  {self.symptoms['additional_notes']}")
        
        return "\n".join(report)


def main():
    """Run ChatReport"""
    try:
        chatbot = ChatReportLite()
        chatbot.run()
    except KeyboardInterrupt:
        print("\n\nChat ended by user.")
    except Exception as e:
        print(f"\n\nError: {e}")


if __name__ == "__main__":
    main()
