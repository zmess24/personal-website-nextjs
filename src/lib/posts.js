import { remark } from "remark";
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkRehype from "remark-rehype";
import rehypeStringify from "rehype-stringify";
import remarkParse from "remark-parse";
import remarkImages from "remark-images";
import remarkGfm from "remark-gfm";
import prism from "remark-prism";
import moment from "moment";
import rehypePostBlocks from "./rehypePostBlocks";

const postsDirectory = path.join(process.cwd(), "src/posts");
const DATE_FORMAT = "MM-DD-YYYY";

// Summaries in front matter end with "..."; the new design shows them as plain lines.
const cleanDescription = (d) => (d || "").replace(/\s*(\.\.\.|…)\s*$/, "");

function readPost(fileName) {
	const id = fileName.replace(/\.md$/, "");
	const fileContents = fs.readFileSync(path.join(postsDirectory, fileName), "utf8");
	const matterResult = matter(fileContents);
	return { id, matterResult };
}

export function getSortedPostsData() {
	const allPostsData = fs
		.readdirSync(postsDirectory)
		.filter((f) => f.endsWith(".md"))
		.map((fileName) => {
			const { id, matterResult } = readPost(fileName);
			return {
				id,
				...matterResult.data,
				description: cleanDescription(matterResult.data.description),
			};
		});
	// Newest first
	return allPostsData.sort((a, b) => moment(b.date, DATE_FORMAT).valueOf() - moment(a.date, DATE_FORMAT).valueOf());
}

export async function getPostData(id) {
	const { matterResult } = readPost(`${id}.md`);
	const toc = [];

	const processedContent = await remark()
		.use(remarkParse)
		.use(remarkImages)
		.use(remarkGfm)
		.use(prism)
		.use(remarkMath)
		.use(remarkRehype)
		.use(rehypeKatex)
		.use(rehypePostBlocks, { toc })
		.use(rehypeStringify)
		.process(matterResult.content);
	const contentHtml = processedContent.toString();

	// Reading time from the prose (code blocks left out), at ~225 words a minute.
	const prose = matterResult.content.replace(/```[\s\S]*?```/g, " ").replace(/^(?: {4}|\t).*$/gm, " ");
	const words = prose.split(/\s+/).filter(Boolean).length;
	const readingMinutes = Math.max(1, Math.round(words / 225));

	return {
		id,
		contentHtml,
		toc,
		readingMinutes,
		...matterResult.data,
		description: cleanDescription(matterResult.data.description),
	};
}

export function getAllPostIds() {
	return fs
		.readdirSync(postsDirectory)
		.filter((f) => f.endsWith(".md"))
		.map((fileName) => ({ params: { id: fileName.replace(/\.md$/, "") } }));
}
