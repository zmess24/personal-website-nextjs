"use client";
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
				<link
					rel="stylesheet"
					href="https://cdnjs.cloudflare.com/ajax/libs/prism-themes/1.9.0/prism-coldark-dark.min.css"
					integrity="sha512-UE88w575S5hQlj3QhY249ZKOe9noZYPtmKL6DwZnKQtTFRCw2dkRfUdp6GwxeV/mig7Q9G7H3vcX8ETVRDRrTg=="
					crossorigin="anonymous"
					referrerpolicy="no-referrer"
				/>
				{/* <link href="https://cdnjs.cloudflare.com/ajax/libs/prism-themes/1.9.0/prism-coy-without-shadows.min.css" rel="stylesheet" /> */}
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
