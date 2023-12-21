import { getAllPostIds, getPostData } from "../../lib/posts";
import Head from "next/head";
import "../../styles/master.scss";
import PostNav from "./components/PostNav";
import moment from "moment";

export default function Post({ postData }) {
	return (
		<main>
			<Head>
				<title>{postData.title}</title>
			</Head>
			<PostNav />
			<div className="post-container">
				<article>
					<section>
						<h1>{postData.title}</h1>
						<span className="date">{moment(postData.date).format("MMM DD YYYY")}</span>
					</section>
					<hr />
					<div dangerouslySetInnerHTML={{ __html: postData.contentHtml }}></div>
				</article>
			</div>
		</main>
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
