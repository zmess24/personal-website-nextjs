import * as React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
// import { GatsbyImage, getImage } from "gatsby-plugin-image";

const Card = ({ data: { frontmatter } }) => {
	// const image = getImage(frontmatter.image);

	return (
		<main className="card" key={frontmatter.title}>
			{/* <section className="card-image">
				<a className="image is-4by3" href={frontmatter.link} target="_blank" rel="noreferrer"></a>
			</section> */}
			<a href={frontmatter.link} target="_blank" rel="noreferrer">
				{/* <GatsbyImage className="card-image " alt="image" image={image} /> */}
			</a>
			<section className="card-content">
				<h6 className="subtitle is-6">{frontmatter.title}</h6>
				<p>{frontmatter.description}</p>
				<ul id="tags" className="is-flex-direction-row">
					{frontmatter.technologies &&
						frontmatter.technologies.map((tag) => {
							return (
								<li key={tag} className="tag is-info is-light">
									{tag}
								</li>
							);
						})}
				</ul>
			</section>
			<section className="card-footer">
				<span className="date">{frontmatter.date}</span>
				<a href={frontmatter.github} target="_blank" rel="noreferrer">
					<FontAwesomeIcon size="lg" icon={["fab", "github"]} />
				</a>
			</section>
		</main>
	);
};

export default Card;
