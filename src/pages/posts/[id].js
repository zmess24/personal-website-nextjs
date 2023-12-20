import { getAllPostIds, getPostData } from "../../lib/posts";

export default function Post({ postData }) {
	console.log(postData);

	return <h1>Hello World!</h1>;
}

export async function getStaticProps({ id }) {
	const postData = await getPostData(id);
	return {
		props: {
			postData,
		},
	};
}

export async function getStaticPaths() {
	const paths = getAllPostIds();
	return {
		paths,
		fallback: false,
	};
}
