import { getAllPostIds, getPostData } from "../../lib/posts";
import Head from "next/head";
import "../../styles/master.scss";

export default function Post({ postData }) {
	return (
		<div className="post-container">
			<Head>
				<title>{postData.title}</title>
			</Head>
			<section>
				<h1>{postData.title}</h1>
				<span className="date">{postData.date}</span>
			</section>
			<hr />
			<article dangerouslySetInnerHTML={{ __html: postData.contentHtml }}></article>
		</div>
	);
}

export async function getStaticProps({ params }) {
	const postData = await getPostData(params.id);

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
