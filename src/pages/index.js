import Head from "next/head";
import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";
import NowCard from "@/components/NowCard";
import Timeline from "@/components/Timeline";
import SwishJam from "@/components/SwishJam";
import { getSortedProjectsData } from "@/lib/projects";
import { getSortedPostsData } from "@/lib/posts";
import { buildEvents } from "@/lib/timeline";
import { BIO, NOW, SITE_URL } from "@/data/site";

export async function getStaticProps() {
	const posts = getSortedPostsData().map(({ id, title, description, date }) => ({ id, title, description, date }));
	const projects = getSortedProjectsData().map(({ title, description, date, link }) => ({ title, description, date, link }));
	return { props: { layout: buildEvents({ posts, projects }) } };
}

export default function Home({ layout }) {
	return (
		<>
			<Head>
				<title>Zachary Messinger</title>
				<meta name="description" content={BIO} />
				<meta property="og:title" content="Zachary Messinger" />
				<meta property="og:description" content={BIO} />
				<meta property="og:url" content={SITE_URL} />
			</Head>
			<SwishJam>
				<div className="wrap">
					<SiteHeader />
					<main id="top">
						<section className="hero">
							<h1>
								Zachary
								<br />
								Messinger
							</h1>
							<div>
								<p>{BIO}</p>
							</div>
						</section>

						<section className="split" id="now">
							{NOW.map((n) => (
								<NowCard key={n.name} n={n} />
							))}
						</section>

						<section className="line" aria-labelledby="line-h">
							<div className="line-h">
								<h2 id="line-h" className="caps">
									Everything since {layout.startYear}
								</h2>
								<div className="legend caps">
									<span>
										<i />
										Writing
									</span>
									<span>
										<i className="p" />
										Built
									</span>
									<span>
										<i className="j" />
										Work
									</span>
								</div>
							</div>
							<Timeline layout={layout} />
						</section>
					</main>
					<SiteFooter />
				</div>
			</SwishJam>
		</>
	);
}
