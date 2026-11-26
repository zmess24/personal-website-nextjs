import Head from "next/head";
import Image from "next/image";
import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";
import PixelMark from "@/components/PixelMark";
import { TimelineRail, TimelineList } from "@/components/Timeline";
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
					<SiteHeader
						links={[
							{ href: "#line", label: "Timeline" },
							{ href: "#now", label: "Now" },
						]}
					/>
					<main id="top">
						<section className="hero">
							<h1>
								Zachary
								<br />
								Messinger
							</h1>
							<div>
								<p>{BIO}</p>
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
						</section>
					</main>
				</div>

				<TimelineRail layout={layout} />

				<div className="wrap">
					<p className="hint caps">Scroll the line &rarr; it ends at what I&apos;m building now</p>
					<TimelineList layout={layout} />

					<section className="split" id="now">
						{NOW.map((n) => (
							<a key={n.name} className="big" href={n.url} target="_blank" rel="noreferrer">
								<span className="mk">
									{n.mark === "clz" ? (
										<PixelMark word="CLZ" label="CLZ Labs" />
									) : (
										<Image src={n.icon} alt="" width={64} height={64} />
									)}
								</span>
								<div>
									<h2>{n.name}</h2>
									<p>{n.description}</p>
									<p className="h caps">
										{n.since} &middot; {n.host} &rarr;
									</p>
								</div>
							</a>
						))}
					</section>

					<SiteFooter />
				</div>
			</SwishJam>
		</>
	);
}
