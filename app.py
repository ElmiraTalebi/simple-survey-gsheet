import random
import time
from datetime import datetime
from typing import Dict, List, Optional

class ChatReport:
    """
    A conversational symptom reporting chatbot for head and neck cancer patients.
    This prototype simulates natural conversation to collect symptom information.
    """
    
    def __init__(self):
        self.symptoms = {}
        self.conversation_history = []
        self.patient_name = None
        self.current_topic = None
        
        # Symptom domains to assess
        self.symptom_domains = {
            'pain': ['location', 'severity', 'frequency', 'management'],
            'mouth': ['dry_mouth', 'sores', 'taste_changes', 'severity'],
            'swallowing': ['difficulty_level', 'types_of_food', 'pain_when_swallowing'],
            'nutrition': ['appetite', 'weight_changes', 'food_intake', 'feeding_tube'],
            'breathing': ['difficulty', 'frequency', 'severity'],
            'mood': ['anxiety', 'depression', 'worry', 'sleep'],
            'energy': ['fatigue_level', 'daily_activities', 'rest_needed']
        }
        
    def add_to_history(self, speaker: str, message: str):
        """Track conversation for report generation"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.conversation_history.append({
            'timestamp': timestamp,
            'speaker': speaker,
            'message': message
        })
    
    def display_message(self, message: str, typing_speed: float = 0.02):
        """Display bot message with typing effect"""
        print("\n🤖 ChatReport:", end=" ")
        for char in message:
            print(char, end='', flush=True)
            time.sleep(typing_speed)
        print()
        self.add_to_history('bot', message)
    
    def get_user_input(self, prompt: str = "") -> str:
        """Get user input"""
        if prompt:
            print(f"\n{prompt}")
        user_input = input("You: ").strip()
        self.add_to_history('user', user_input)
        return user_input
    
    def greeting(self):
        """Initial greeting"""
        self.display_message(
            "Hello! I'm ChatReport, here to help you share information about "
            "how you've been feeling before your upcoming appointment."
        )
        time.sleep(0.5)
        self.display_message(
            "This should only take about 10-15 minutes. You can take your time, "
            "and please feel free to share as much or as little detail as you'd like."
        )
        time.sleep(0.5)
        self.display_message("What's your first name?")
        
        name = self.get_user_input()
        self.patient_name = name
        
        self.display_message(
            f"Nice to meet you, {name}! Thank you for taking the time to do this. "
            f"Let's start with some general questions."
        )
    
    def assess_pain(self):
        """Conversational pain assessment"""
        self.current_topic = 'pain'
        self.display_message(
            "Let's talk about any pain you might be experiencing. "
            "Have you had any pain since your last appointment?"
        )
        
        response = self.get_user_input().lower()
        
        if any(word in response for word in ['no', 'not', 'none', "haven't"]):
            self.symptoms['pain'] = {'present': False}
            self.display_message("That's good to hear. We'll move on to the next topic.")
            return
        
        # Pain is present - gather details
        self.symptoms['pain'] = {'present': True, 'details': {}}
        
        self.display_message("I see. Can you tell me where you're feeling the pain?")
        location = self.get_user_input()
        self.symptoms['pain']['details']['location'] = location
        
        self.display_message(
            "On a scale of 0 to 10, where 0 is no pain and 10 is the worst pain imaginable, "
            "how would you rate your pain at its worst?"
        )
        severity = self.get_user_input()
        self.symptoms['pain']['details']['severity'] = severity
        
        self.display_message("How often would you say you experience this pain?")
        frequency = self.get_user_input()
        self.symptoms['pain']['details']['frequency'] = frequency
        
        self.display_message(
            "Are you doing anything to manage the pain? For example, taking medication, "
            "using ice, or anything else?"
        )
        management = self.get_user_input()
        self.symptoms['pain']['details']['management'] = management
        
        # Empathetic follow-up
        if any(num in severity for num in ['7', '8', '9', '10']):
            self.display_message(
                "I'm sorry you're experiencing such significant pain. "
                "Your doctor will definitely want to discuss this with you."
            )
    
    def assess_mouth_symptoms(self):
        """Assess mouth-related symptoms"""
        self.current_topic = 'mouth'
        self.display_message(
            "Now let's talk about your mouth. Many patients experience dry mouth, "
            "mouth sores, or changes in taste during treatment. "
            "Have you noticed any of these issues?"
        )
        
        response = self.get_user_input().lower()
        
        if any(word in response for word in ['no', 'not', 'none']):
            self.symptoms['mouth'] = {'present': False}
            self.display_message("Great! Let's continue.")
            return
        
        self.symptoms['mouth'] = {'present': True, 'details': {}}
        
        # Identify specific issues
        self.display_message(
            "Could you describe what you're experiencing? For example, "
            "is your mouth feeling dry, do you have sores, or has your sense of taste changed?"
        )
        description = self.get_user_input()
        self.symptoms['mouth']['details']['description'] = description
        
        self.display_message(
            "How much is this affecting your daily life? Would you say it's mild, "
            "moderate, or severe?"
        )
        severity = self.get_user_input()
        self.symptoms['mouth']['details']['severity'] = severity
    
    def assess_swallowing(self):
        """Assess swallowing difficulties"""
        self.current_topic = 'swallowing'
        self.display_message(
            "Let's talk about eating and swallowing. Have you had any difficulty "
            "swallowing since your last visit?"
        )
        
        response = self.get_user_input().lower()
        
        if any(word in response for word in ['no', 'not', 'none', "haven't", 'easy']):
            self.symptoms['swallowing'] = {'present': False}
            self.display_message("That's good. Let's move forward.")
            return
        
        self.symptoms['swallowing'] = {'present': True, 'details': {}}
        
        self.display_message(
            "Can you tell me more about the difficulty? For instance, "
            "is it painful, or does food feel like it's getting stuck?"
        )
        description = self.get_user_input()
        self.symptoms['swallowing']['details']['description'] = description
        
        self.display_message(
            "What types of foods are you able to eat right now? "
            "For example, are you eating soft foods, liquids, or regular foods?"
        )
        food_types = self.get_user_input()
        self.symptoms['swallowing']['details']['food_types'] = food_types
    
    def assess_nutrition(self):
        """Assess nutritional intake"""
        self.current_topic = 'nutrition'
        self.display_message(
            "I'd like to ask about your eating and nutrition. "
            "How has your appetite been lately?"
        )
        
        appetite = self.get_user_input()
        self.symptoms['nutrition'] = {'details': {'appetite': appetite}}
        
        self.display_message(
            "Have you noticed any changes in your weight? "
            "For example, have you lost or gained weight?"
        )
        weight = self.get_user_input()
        self.symptoms['nutrition']['details']['weight_changes'] = weight
        
        # Check for concerning weight loss
        if any(word in weight.lower() for word in ['lost', 'losing', 'dropped', 'down']):
            self.display_message(
                "I see. Do you happen to know approximately how much weight you've lost?"
            )
            amount = self.get_user_input()
            self.symptoms['nutrition']['details']['weight_amount'] = amount
    
    def assess_mood(self):
        """Assess emotional wellbeing"""
        self.current_topic = 'mood'
        self.display_message(
            "I want to check in on how you're doing emotionally. "
            "Treatment can be challenging, and it's normal to have ups and downs. "
            "How would you describe your mood lately?"
        )
        
        mood = self.get_user_input().lower()
        self.symptoms['mood'] = {'details': {'description': mood}}
        
        # Check for concerning responses
        if any(word in mood for word in ['worried', 'anxious', 'depressed', 'sad', 'scared', 'down', 'hopeless']):
            self.display_message(
                "Thank you for sharing that with me. Many patients feel this way, "
                "and your care team wants to support you. "
                "Are you experiencing any trouble sleeping?"
            )
            sleep = self.get_user_input()
            self.symptoms['mood']['details']['sleep'] = sleep
            
            self.display_message(
                "Your doctor will want to discuss ways to help you feel better. "
                "You're not alone in this."
            )
    
    def generate_report(self) -> str:
        """Generate a clinical report from collected information"""
        report = []
        report.append("=" * 60)
        report.append("CHATREPORT SYMPTOM SUMMARY")
        report.append("=" * 60)
        report.append(f"Patient Name: {self.patient_name}")
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 60)
        report.append("")
        
        # Pain
        if self.symptoms.get('pain', {}).get('present'):
            report.append("🔴 PAIN - PRESENT")
            details = self.symptoms['pain']['details']
            report.append(f"  Location: {details.get('location', 'Not specified')}")
            report.append(f"  Severity: {details.get('severity', 'Not specified')}/10")
            report.append(f"  Frequency: {details.get('frequency', 'Not specified')}")
            report.append(f"  Management: {details.get('management', 'Not specified')}")
        else:
            report.append("✅ PAIN - Not reported")
        report.append("")
        
        # Mouth symptoms
        if self.symptoms.get('mouth', {}).get('present'):
            report.append("🔴 MOUTH SYMPTOMS - PRESENT")
            details = self.symptoms['mouth']['details']
            report.append(f"  Description: {details.get('description', 'Not specified')}")
            report.append(f"  Severity: {details.get('severity', 'Not specified')}")
        else:
            report.append("✅ MOUTH SYMPTOMS - Not reported")
        report.append("")
        
        # Swallowing
        if self.symptoms.get('swallowing', {}).get('present'):
            report.append("🔴 SWALLOWING DIFFICULTY - PRESENT")
            details = self.symptoms['swallowing']['details']
            report.append(f"  Description: {details.get('description', 'Not specified')}")
            report.append(f"  Food Types: {details.get('food_types', 'Not specified')}")
        else:
            report.append("✅ SWALLOWING - No difficulties reported")
        report.append("")
        
        # Nutrition
        if 'nutrition' in self.symptoms:
            report.append("📊 NUTRITION STATUS")
            details = self.symptoms['nutrition']['details']
            report.append(f"  Appetite: {details.get('appetite', 'Not specified')}")
            report.append(f"  Weight Changes: {details.get('weight_changes', 'Not specified')}")
            if 'weight_amount' in details:
                report.append(f"  Amount: {details['weight_amount']}")
        report.append("")
        
        # Mood
        if 'mood' in self.symptoms:
            report.append("💭 EMOTIONAL WELLBEING")
            details = self.symptoms['mood']['details']
            report.append(f"  Mood: {details.get('description', 'Not specified')}")
            if 'sleep' in details:
                report.append(f"  Sleep: {details['sleep']}")
        
        report.append("")
        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def closing(self):
        """Closing message"""
        self.display_message(
            f"Thank you so much for taking the time to share this information with me, {self.patient_name}. "
        )
        time.sleep(0.5)
        self.display_message(
            "I've created a summary report that your doctor will review before your appointment. "
            "This will help them better understand how you've been doing."
        )
        time.sleep(0.5)
        self.display_message(
            "Is there anything else you'd like to add that we haven't covered?"
        )
        
        additional = self.get_user_input()
        
        if additional and additional.lower() not in ['no', 'nope', 'nothing', 'no thanks']:
            self.symptoms['additional_notes'] = additional
            self.display_message(
                "Thank you for sharing that. I've added it to your report."
            )
        
        self.display_message(
            "Take care, and we'll see you at your upcoming appointment!"
        )
    
    def run(self):
        """Main conversation flow"""
        print("\n" + "="*60)
        print("  CHATREPORT - Symptom Reporting System")
        print("  Fox Chase Cancer Center")
        print("="*60)
        
        time.sleep(1)
        
        try:
            # Greeting
            self.greeting()
            time.sleep(1)
            
            # Symptom assessments
            self.assess_pain()
            time.sleep(1)
            
            self.assess_mouth_symptoms()
            time.sleep(1)
            
            self.assess_swallowing()
            time.sleep(1)
            
            self.assess_nutrition()
            time.sleep(1)
            
            self.assess_mood()
            time.sleep(1)
            
            # Closing
            self.closing()
            
            # Generate and display report
            print("\n\n")
            print(self.generate_report())
            
        except KeyboardInterrupt:
            print("\n\nChat ended by user.")
        except Exception as e:
            print(f"\n\nAn error occurred: {e}")


def main():
    """Run the ChatReport demo"""
    chatbot = ChatReport()
    chatbot.run()


if __name__ == "__main__":
    main()
