import { getAllPostIds } from "../../lib/posts";

export async function getStaticPaths() {
	const paths = getAllPostIds();
	return {
		paths,
		fallback: false,
	};
}

export default function Posts() {
	return <h1>This is a post!</h1>;
}
