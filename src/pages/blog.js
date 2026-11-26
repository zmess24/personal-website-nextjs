import Head from "next/head";
import Link from "next/link";
import moment from "moment";
import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";
import SwishJam from "@/components/SwishJam";
import { getSortedPostsData } from "@/lib/posts";
import { SITE_URL } from "@/data/site";

export async function getStaticProps() {
	const posts = getSortedPostsData().map(({ id, title, description, date, tags, readingMinutes }) => ({
		id,
		title,
		description,
		tags: tags || [],
		readingMinutes,
		day: moment(date, "MM-DD-YYYY").format("MMM D"),
		year: moment(date, "MM-DD-YYYY").year(),
	}));
	const years = [];
	posts.forEach((p) => {
		if (!years.length || years[years.length - 1].year !== p.year) years.push({ year: p.year, posts: [] });
		years[years.length - 1].posts.push(p);
	});
	return { props: { years, count: posts.length } };
}

export default function Blog({ years, count }) {
	return (
		<>
			<Head>
				<title>Blog · Zachary Messinger</title>
				<meta name="description" content="Writing on AI and machine learning, built from scratch and written up." />
				<meta property="og:title" content="Blog · Zachary Messinger" />
				<meta property="og:url" content={`${SITE_URL}/blog`} />
			</Head>
			<SwishJam>
				<div className="wrap">
					<SiteHeader />
					<main>
						<header className="page-h">
							<h1>Blog</h1>
							<p>
								{count} posts on AI and machine learning. I build each idea from scratch, then write up how it works.
							</p>
						</header>
						<div className="idx">
							{years.map(({ year, posts }) => (
								<section key={year} className="idx-y" aria-labelledby={`y-${year}`}>
									<h2 id={`y-${year}`}>{year}</h2>
									<ol>
										{posts.map((p) => (
											<li key={p.id}>
												<Link href={`/posts/${p.id}`}>
													<span className="d">{p.day}</span>
													<span className="b">
														<span className="tt">{p.title}</span>
														{p.description && <span className="su">{p.description}</span>}
													</span>
													<span className="m">
														{p.tags[0] && <span>{p.tags[0]}</span>}
														<span>{p.readingMinutes} min</span>
													</span>
												</Link>
											</li>
										))}
									</ol>
								</section>
							))}
						</div>
					</main>
					<SiteFooter />
				</div>
			</SwishJam>
		</>
	);
}
