import fs from "fs";
import path from "path";
import matter from "gray-matter";
import moment from "moment";

const projectsDirectory = path.join(process.cwd(), "src/projects");

export function getSortedProjectsData() {
	// Get file names under /posts
	const fileNames = fs.readdirSync(projectsDirectory);
	const allProjectData = fileNames.map((fileName) => {
		// Remove ".md" from file name to get id
		const id = fileName.replace(/\.md$/, "");

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
	// Sort posts by date
	return allProjectData.sort((a, b) => {
		a = moment(a.date);
		b = moment(b.date);
		debugger;
		if (a < b) {
			return 1;
		} else {
			return -1;
		}
	});
}
