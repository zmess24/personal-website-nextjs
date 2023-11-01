import * as React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faGithub } from "@fortawesome/free-brands-svg-icons";
import Link from "next/link";

const Card = ({ data }) => {
	return (
		<main className="card" key={data.title}>
			<Link href={data.link} target="_blank" rel="noreferrer">
				<img className="card-image " alt="image" src={data.image} />
			</Link>
			<section className="card-content">
				<h6 className="subtitle is-6">{data.title}</h6>
				<p>{data.description}</p>
				<ul id="tags" className="is-flex-direction-row">
					{data.technologies &&
						data.technologies.map((tag) => {
							return (
								<li key={tag} className="tag is-info is-light">
									{tag}
								</li>
							);
						})}
				</ul>
			</section>
			<section className="card-footer">
				<span className="date">{data.date}</span>
				<Link href={data.github} target="_blank" rel="noreferrer">
					<FontAwesomeIcon size="lg" icon={faGithub} />
				</Link>
			</section>
		</main>
	);
};

export default Card;
