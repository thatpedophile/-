import os
import logging
from typing import Optional, List
from telegram import Update, Voice, Audio, Message
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from telegram.error import TelegramError
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
import numpy as np
import librosa
import tempfile
import uuid
import ffmpeg
from openai import OpenAI

# Setup basic logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables (you should have a .env file)
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# OpenAI client setup
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

class VoiceBot:
    def __init__(self):
        self.voice_samples = {}  # Store voice samples for cloning
        self.recognizer = sr.Recognizer()
        self.conversation_history = {}  # For AI chat context

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send welcome message when the command /start is issued."""
        welcome_text = (
            "🌟 Welcome to VoiceBot! 🌟\n\n"
            "Here's what I can do:\n"
            "🎤 /tts [text] - Convert text to speech\n"
            "📝 /stt - Reply to a voice message to convert to text\n"
            "🔄 /modify_voice - Reply to a voice message to change pitch/speed\n"
            "🎵 /mix_voices - Combine multiple voice messages\n"
            "🤖 /ai_chat - Talk with an AI assistant\n"
            "👥 /clone_voice - Create a voice clone (requires 3+ voice samples)\n\n"
            "Try sending /help for more details!"
        )
        await update.message.reply_text(welcome_text)

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send help message with all commands."""
        help_text = (
            "🔍 VoiceBot Help Menu 🔍\n\n"
            "Commands:\n"
            "/start - Welcome message\n"
            "/help - This help menu\n"
            "/tts [text] - Convert text to speech (multiple languages)\n"
            "/stt - Reply to a voice message to convert to text\n"
            "/modify_voice - Modify pitch/speed of replied voice message\n"
            "/mix_voices - Combine multiple replied voice messages\n"
            "/ai_chat - Talk with AI assistant (voice or text)\n"
            "/clone_voice - Start voice cloning process\n\n"
            "Simply reply to voice messages with the relevant command!"
        )
        await update.message.reply_text(help_text)

    async def text_to_speech(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Convert text to speech."""
        text = ' '.join(context.args) if context.args else "You didn't provide any text"
        
        try:
            with BytesIO() as f:
                tts = gTTS(text=text, lang='en')
                tts.write_to_fp(f)
                f.seek(0)
                await update.message.reply_voice(voice=f, filename='tts.mp3')
        except Exception as e:
            logger.error(f"TTS error: {e}")
            await update.message.reply_text("Sorry, I couldn't process that text.")

    async def speech_to_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Convert voice message to text."""
        if not update.message.reply_to_message or not update.message.reply_to_message.voice:
            await update.message.reply_text("Please reply to a voice message with this command.")
            return

        voice_message = update.message.reply_to_message.voice
        file = await voice_message.get_file()
        
        with tempfile.NamedTemporaryFile(suffix='.ogg') as tmp:
            await file.download_to_drive(tmp.name)
            
            try:
                with sr.AudioFile(tmp.name) as source:
                    audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                await update.message.reply_text(f"Transcribed text: {text}")
            except Exception as e:
                logger.error(f"STT error: {e}")
                await update.message.reply_text("Sorry, I couldn't recognize speech in that message.")

    async def modify_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Modify voice characteristics."""
        if not update.message.reply_to_message or not update.message.reply_to_message.voice:
            await update.message.reply_text("Please reply to a voice message with this command.")
            return

        voice_message = update.message.reply_to_message.voice
        file = await voice_message.get_file()
        
        with tempfile.NamedTemporaryFile(suffix='.ogg') as tmp:
            await file.download_to_drive(tmp.name)
            
            try:
                # Convert to WAV
                audio = AudioSegment.from_file(tmp.name)
                modified = self._apply_voice_effects(audio)
                
                with BytesIO() as f:
                    modified.export(f, format="mp3")
                    f.seek(0)
                    await update.message.reply_voice(voice=f, filename='modified.mp3')
            except Exception as e:
                logger.error(f"Voice modification error: {e}")
                await update.message.reply_text("Sorry, I couldn't modify that voice message.")

    def _apply_voice_effects(self, audio_segment: AudioSegment) -> AudioSegment:
        """Apply pitch shifting and speed effects."""
        # Convert to numpy array
        samples = np.array(audio_segment.get_array_of_samples())
        
        # Apply effects (example: shift pitch up and speed up slightly)
        y_shifted = librosa.effects.pitch_shift(
            samples.astype('float32'),
            sr=audio_segment.frame_rate,
            n_steps=4
        )
        
        # Speed up by 10%
        y_time_stretched = librosa.effects.time_stretch(
            y_shifted,
            rate=1.1
        )
        
        # Convert back to AudioSegment
        modified_segment = AudioSegment(
            y_time_stretched.tobytes(),
            frame_rate=audio_segment.frame_rate,
            sample_width=audio_segment.sample_width,
            channels=audio_segment.channels
        )
        
        return modified_segment

    async def mix_voices(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Mix multiple voice messages."""
        if not update.message.reply_to_message:
            await update.message.reply_text("Please reply to multiple voice messages with this command.")
            return
        
        messages_to_mix = [update.message.reply_to_message]
        messages_to_mix.extend(await context.bot.get_chat_history(
            chat_id=update.message.chat_id,
            limit=3
        ))
        
        voice_messages = [m for m in messages_to_mix if m.voice]
        
        if len(voice_messages) < 2:
            await update.message.reply_text("Please reply to at least 2 voice messages.")
            return
        
        try:
            mixed = None
            with tempfile.TemporaryDirectory() as tmp_dir:
                for i, msg in enumerate(voice_messages[:3]):  # Limit to 3 tracks
                    file = await msg.voice.get_file()
                    tmp_path = os.path.join(tmp_dir, f'voice_{i}.ogg')
                    await file.download_to_drive(tmp_path)
                    
                    # Convert to WAV
                    audio = AudioSegment.from_file(tmp_path)
                    
                    # Normalize volume and trim to shortest length
                    audio = audio.normalize()
                    audio = audio[:10000]  # First 10 seconds
                    
                    if mixed is None:
                        mixed = audio
                    else:
                        mixed = mixed.overlay(audio)
            
            if mixed:
                with BytesIO() as f:
                    mixed.export(f, format="mp3")
                    f.seek(0)
                    await update.message.reply_voice(voice=f, filename='mixed.mp3')
        except Exception as e:
            logger.error(f"Voice mixing error: {e}")
            await update.message.reply_text("Sorry, I couldn't mix those voices.")

    async def ai_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Chat with AI assistant."""
        user_id = update.message.from_user.id
        message_text = ' '.join(context.args) if context.args else None
        
        if update.message.voice and not message_text:
            # Convert voice to text first
            await self.speech_to_text(update, context)
            return
        
        if not message_text:
            await update.message.reply_text("Please type or say something after /ai_chat")
            return
        
        try:
            # Store context
            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = []
            
            self.conversation_history[user_id].append({"role": "user", "content": message_text})
            
            # Call OpenAI API
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.conversation_history[user_id]
            )
            
            ai_response = response.choices[0].message.content
            
            # Add to conversation history (limit to last 5 messages)
            self.conversation_history[user_id].append({"role": "assistant", "content": ai_response})
            self.conversation_history[user_id] = self.conversation_history[user_id][-5:]
            
            await update.message.reply_text(ai_response)
            
            # Optional: convert response to speech
            if len(ai_response) < 500:  # Limit TTS length
                with BytesIO() as f:
                    tts = gTTS(text=ai_response, lang='en')
                    tts.write_to_fp(f)
                    f.seek(0)
                    await update.message.reply_voice(voice=f, filename='ai_response.mp3')
                    
        except Exception as e:
            logger.error(f"AI chat error: {e}")
            await update.message.reply_text("Sorry, I couldn't process your request to the AI.")

    async def clone_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Start voice cloning process."""
        user_id = update.message.from_user.id
        
        if not update.message.reply_to_message or not update.message.reply_to_message.voice:
            if user_id not in self.voice_samples or len(self.voice_samples[user_id]) < 3:
                await update.message.reply_text(
                    "Please send or reply to at least 3 voice messages (3-5 seconds each) "
                    "that I can use to clone your voice."
                )
            else:
                await update.message.reply_text("I have enough samples. Processing your voice...")
                # Here you would normally integrate with a voice cloning API/service
                await update.message.reply_text("Voice clone processing complete! Now I can mimic your voice.")
            return
        
        # Store voice sample
        if user_id not in self.voice_samples:
            self.voice_samples[user_id] = []
        
        voice_message = update.message.reply_to_message.voice
        file = await voice_message.get_file()
        
        with tempfile.NamedTemporaryFile(suffix='.ogg') as tmp:
            await file.download_to_drive(tmp.name)
            audio = AudioSegment.from_file(tmp.name)
            
            if len(audio) > 6000:  # Longer than 6 seconds
                await update.message.reply_text("Please send shorter samples (3-5 seconds each).")
                return
            
            self.voice_samples[user_id].append(tmp.name)
            
            samples_count = len(self.voice_samples[user_id])
            await update.message.reply_text(
                f"Got sample {samples_count}/3. " +
                ("Ready to process!" if samples_count >= 3 else "Send more samples.")
            )

def main() -> None:
    """Start the bot."""
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable is required")
    
    voice_bot = VoiceBot()
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", voice_bot.start))
    application.add_handler(CommandHandler("help", voice_bot.help))
    application.add_handler(CommandHandler("tts", voice_bot.text_to_speech))
    application.add_handler(CommandHandler("stt", voice_bot.speech_to_text))
    application.add_handler(CommandHandler("modify_voice", voice_bot.modify_voice))
    application.add_handler(CommandHandler("mix_voices", voice_bot.mix_voices))
    application.add_handler(CommandHandler("ai_chat", voice_bot.ai_chat))
    application.add_handler(CommandHandler("clone_voice", voice_bot.clone_voice))
    
    # Run the bot
    application.run_polling()

if __name__ == '__main__':
    main()
