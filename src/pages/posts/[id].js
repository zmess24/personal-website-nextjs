"use client";
import { getAllPostIds, getPostData } from "../../lib/posts";
import Head from "next/head";
import "../../styles/master.scss";
import PostNav from "./components/PostNav";
import moment from "moment";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faClock, faCalendar } from "@fortawesome/free-regular-svg-icons";
import { faTags } from "@fortawesome/free-solid-svg-icons";
import SwishJam from "@/components/SwishJam";

export default function Post({ postData }) {
	let minutes = Math.round(postData.contentHtml.split(" ").length / 700);

	return (
		<>
			<Head>
				<title>{postData.title}</title>
				<link
					rel="stylesheet"
					href="https://cdnjs.cloudflare.com/ajax/libs/prism-themes/1.9.0/prism-coldark-dark.min.css"
					integrity="sha512-UE88w575S5hQlj3QhY249ZKOe9noZYPtmKL6DwZnKQtTFRCw2dkRfUdp6GwxeV/mig7Q9G7H3vcX8ETVRDRrTg=="
					crossOrigin="anonymous"
					referrerPolicy="no-referrer"
				/>
				<link href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css" rel="stylesheet" />
			</Head>
			<SwishJam>
				<main className="post-wrapper">
					<PostNav />
					<div className="post-container">
						<article>
							<section>
								<h1>{postData.title}</h1>
								<div className="post-header">
									<span className="item">
										<FontAwesomeIcon size="sm" icon={faCalendar} />
										{moment(postData.date).format("MMM DD YYYY")}
									</span>
									<span className="item">
										<FontAwesomeIcon size="lg" icon={faClock} />
										{minutes} minute read
									</span>
									<span className="item">
										<FontAwesomeIcon size="lg" icon={faTags} />
										<span className="tag is-warning is-light">{postData.tags[0]}</span>
									</span>
								</div>
							</section>
							<hr />
							<div dangerouslySetInnerHTML={{ __html: postData.contentHtml }}></div>
						</article>
					</div>
				</main>
			</SwishJam>
		</>
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
