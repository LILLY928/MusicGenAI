import os
import sys
import io
import argparse
import shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from gradio_client import Client, handle_file
from huggingface_hub import HfApi
from openai import OpenAI
from dotenv import load_dotenv
import shutil


#get the api keys from google drive
load_dotenv(dotenv_path="/mnt/mediaefs/.env")
openAI_api_key = os.getenv("OPENAI_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
client = OpenAI(api_key=openAI_api_key)

#download one image from google drive and submit to huggingface
def faces_gdrive_to_hf():

    #obtain one artist face image from the face folder, then delete the image after use
    #extract the artist name
    faces_dir = "/mnt/mediaefs/faces"
    found_image = None

    for filename in os.listdir(faces_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            found_image = filename
            break

    if found_image:
        src_path = os.path.join(faces_dir, found_image)
        dest_path = os.path.join(".", found_image)

        shutil.copy(src_path, dest_path)
        os.remove(src_path)

        print(f"Downloaded and deleted: {found_image}")
    else:
        print("No image found in faces folder.")

    #upload iamge to huggingface
    api = HfApi()
    api.upload_file(
        path_or_fileobj=dest_path,
        path_in_repo=found_image,
        repo_id="yijin928/ComfyUI-dataset",
        repo_type="dataset",
        token=huggingface_api_key
    )
    artist=found_image[:-1]
    return f"https://huggingface.co/datasets/yijin928/ComfyUI-dataset/resolve/main/{found_image}", artist


#generate a rednote post from the artist name, generate a prompt from the artist name
def generate_post_and_prompt(artist_name):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一位擅长创作小红书爆款内容的专家，专门撰写音乐人介绍类文章。请遵循以下步骤进行创作，以吸引用户并提升互动率。\n\n写作要求\n目标：以生动、有趣、引人入胜的方式介绍音乐人及其作品，激发用户的兴趣和讨论。\n\n风格：内容要轻松有趣、口语化，避免生硬的介绍，像和朋友聊天一样。\n\n互动性：鼓励读者留言、分享或参与讨论，让他们对该音乐人产生兴趣。\n\n内容结构\n\n标题要求\n请根据以下技巧生成 1 个富有吸引力的标题，并适当加入 emoji 表情：\n二极管标题法（对比、冲突、反差感）\n热门关键词（从流行关键词列表随机选 1-2 个）\n制造悬念或好奇心（让读者想点进来一探究竟）\n符合小红书风格，简短、有趣、抓人眼球\n使用音乐人昵称或中文名（让读者更容易共鸣）\n\n正文写作技巧\n先将接收到的英文名翻译成中文，优先使用昵称，如果没有，则使用翻译后的中文名。\n正文需包含 3-5 个自然段，每段需配合 emoji 突出重点，遵循以下原则：\n开篇吸引人：用一句有趣、神秘或情感共鸣的话引入，勾起读者兴趣。\n音乐人介绍：突出其特色、代表作、风格，避免流水账式的描述。\n制造情绪共鸣：可结合故事、听众评价或音乐背后的故事。\n互动引导：结尾引导读者留言，比如询问他们的听歌感受或推荐其他类似的音乐人。\n使用爆款关键词：确保内容符合 SEO，增强曝光度。\n标签优化：从文章中提取 3-6 个高相关度的关键词生成 #标签，并放在文末。\n\n最终输出格式\n请按以下格式输出，不包含额外解释：\n标题\n[标题]\n[换行]\n正文\n[正文]\n标签：[标签]"},
            {"role": "user", "content": "Taylor Swift"},
            {"role": "assistant", "content": "🎤霉霉的故事：从乡村小花到国际巨星✨\n\n🎤提到霉霉，你会想到她哪首歌呢？🎵从《Love Story》到《Blank Space》，她的音乐陪伴了无数人的成长。\n\n🎶霉霉不仅是位卓越的歌手，还是一位不断突破自我的创作天才。她的每一次转型都掀起新的热潮，从乡村到流行，毫无违和感🎸。\n\n🌟她的歌词总是充满故事性，让人听了忍不住细细品味✍️。无论是爱情还是友谊，她的作品都能引发共鸣，仿佛在诉说我们的故事。\n\n💖舞台上的她光芒四射，舞台下，她也在用行动支持公益事业，传递正能量。她的每一次发声都充满了力量和温暖🌈。\n\n🤔你最喜欢霉霉哪一张专辑呢？欢迎留言分享你的听歌心情，也可以推荐其他你喜欢的音乐人哦🎧！\n\n标签：#霉霉 #TaylorSwift #音乐故事 #流行巨星 #情感共鸣"},
            {"role": "user", "content": artist_name}
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1
    )
    post_text=response.choices[0].message.content
    print(post_text)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful prompt generator. When given an artist's name, output a single-scene video prompt.\n\nThe video follows the artist in a music-related activity in their genre. Use simple, direct language, don't expose the artist's name in the prompt."
            },
            {
                "role": "user",
                "content": "Ludwig van Beethoven"
            },
            {
                "role": "assistant",
                "content": "a man of intense focus, is seated at the piano, slightly moving with the music, his hands dancing on the keys. The surrounding curtains, rich and opulent."
            },
            {
                "role": "user",
                "content": "Paul McCartney"
            },
            {
                "role": "assistant",
                "content": "A young man with shoulder-length black hair, wearing a stylish black outfit, playing an acoustic guitar on a dimly lit stage. His full face is visible, showing a calm and focused expression as he strums the guitar. A microphone stand is positioned near him, and a music stand with sheet music is in front of him. "
            },
            {
                "role": "user",
                "content": artist_name
            }
        ],
        temperature=1,
        max_tokens=150,
        top_p=1
    )
    prompt=response.choices[0].message.content
    print(prompt)
    return post_text, prompt  


# Call the ComfyUI workflow set up on HuggingFace Space(https://huggingface.co/spaces/yijin928/Test).
# 1. a prompt for video generation
# 2. an image for face swap
# 3. number of frames (affects the process time, the GPU usage contraint is 25 mins.)

def generate_video(prompt, num_of_frames, link):
    client = Client("yijin928/Test", hf_token=huggingface_api_key)
    result = client.predict(
            positive_prompt=prompt,
            num_frames=num_of_frames,
            input_image=handle_file(link),
            api_name="/generate_video",
            )
    print(result)


    # Extract paths
    video_path = result[0]["video"]
    thumbnail_path = result[1]

    output_dir = "/mnt/mediaefs"
    os.makedirs(output_dir, exist_ok=True)

    shutil.copy(video_path, os.path.join(output_dir, "output_video.mp4"))
    shutil.copy(thumbnail_path, os.path.join(output_dir, "output_preview.png"))

    print("Output saved to /mnt/mediaefs")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Xiaohongshu-style post and video for a music artist.")
    parser.add_argument("--frames", type=int, default=20, help="Number of video frames to generate")
    args = parser.parse_args()

    link,artist=faces_gdrive_to_hf()
    print(artist,link)
    post_text, prompt = generate_post_and_prompt(artist)
    generate_video(prompt, num_of_frames=args.frames, link=link)

    with open("/mnt/mediaefs/rednote_post.txt", "w", encoding="utf-8") as f:
        f.write(post_text+"\n")