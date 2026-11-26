import fs from "fs";
import path from "path";
import matter from "gray-matter";
import moment from "moment";

const DATE_FORMATS = ["MM/DD/YYYY", "MM-DD-YYYY"];
const projectsDirectory = path.join(process.cwd(), "src/projects");

export function getSortedProjectsData() {
	// Get file names under /posts
	const fileNames = fs.readdirSync(projectsDirectory);
	const allProjectData = fileNames.map((fileName) => {
		// Remove ".md" from file name to get id
		const id = fileName.replace(/\.mdx?$/, "");

		// Read markdown file as string
		const fullPath = path.join(projectsDirectory, fileName);
		const fileContents = fs.readFileSync(fullPath, "utf8");

		// Use gray-matter to parse the post metadata section
		const matterResult = matter(fileContents);

		// Combine the data with the id
		return {
			id,
			...matterResult.data,
		};
	});
	// Newest first
	return allProjectData.sort((a, b) => moment(b.date, DATE_FORMATS).valueOf() - moment(a.date, DATE_FORMATS).valueOf());
}
